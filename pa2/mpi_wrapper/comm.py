from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        Your implementation should respect the passed reduction operator.
        The required operators for this assignment are MPI.MIN, MPI.SUM,
        and MPI.MAX.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        rank = self.Get_rank()
        num_procs = self.Get_size()

        # Map the MPI reduction operator to the matching NumPy elementwise op.
        if op == MPI.SUM:
            reduce_fn = np.add
        elif op == MPI.MIN:
            reduce_fn = np.minimum
        elif op == MPI.MAX:
            reduce_fn = np.maximum
        else:
            raise NotImplementedError("Only MPI.SUM, MPI.MIN, and MPI.MAX are implemented in myAllreduce.")

        # Start with this rank's local contribution as the partial reduction.
        # After the communication rounds, dest_array will contain the full result.
        np.copyto(dest_array, src_array)

        # Recursive doubling works directly when the communicator size is a
        # power of two. Each round doubles the number of ranks included in
        # dest_array's partial reduction.
        if num_procs & (num_procs - 1) == 0:
            temp_array = np.empty_like(src_array)
            mask = 1
            while mask < num_procs:
                # Flip one bit of rank to pick the partner for this round:
                # mask=1 pairs (0,1), (2,3), ...
                # mask=2 pairs (0,2), (1,3), ...
                # mask=4 pairs (0,4), (1,5), ...
                peer = rank ^ mask
                # Send the current partial result and receive the peer's
                # partial result in one matched point-to-point call.
                self.comm.Sendrecv(dest_array, dest=peer, recvbuf=temp_array, source=peer)
                # Merge the peer's partial result into our partial result.
                reduce_fn(dest_array, temp_array, out=dest_array)
                mask <<= 1
        # Fallback for non-power-of-two sizes: reduce everything to rank 0,
        # then send the final reduced result back out to the other ranks.
        elif rank == 0:
            temp_array = np.empty_like(src_array)
            for i in range(1, num_procs):
                self.comm.Recv(temp_array, source=i)
                reduce_fn(dest_array, temp_array, out=dest_array)
            for i in range(1, num_procs):
                self.comm.Send(dest_array, dest=i)
        else:
            self.comm.Send(src_array, dest=0)
            self.comm.Recv(dest_array, source=0)
        
        # Track bytes transferred by this rank for the chosen algorithm.
        src_array_byte = src_array.itemsize * src_array.size
        if num_procs & (num_procs - 1) == 0:
            # Each recursive-doubling round sends and receives one full buffer.
            self.total_bytes_transferred += src_array_byte * 2 * (num_procs.bit_length() - 1)
        elif rank == 0:
            self.total_bytes_transferred += src_array_byte * 2 * (num_procs - 1)
        else:
            self.total_bytes_transferred += src_array_byte * 2

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over destination ranks:
          - The local segment is copied directly.
          - For every other destination rank, Sendrecv exchanges the segment
            intended for that rank with the segment that rank sends back.
            
        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.Get_rank()

        array_size = src_array.size
        assert array_size % nprocs == 0, "src_array size must be divisible by the number of processes"
        assert dest_array.size == array_size, "dest_array must be the same size as src_array"

        segment_size = array_size // nprocs
        if segment_size == 1:
            # Fast path for the assignment benchmark: each rank sends one value
            # to every peer. Post all point-to-point operations before waiting.
            dest_array[rank:rank + 1] = src_array[rank:rank + 1]
            requests = []
            for peer in range(nprocs):
                if peer != rank:
                    requests.append(self.comm.Irecv(dest_array[peer:peer + 1], source=peer))
            for peer in range(nprocs):
                if peer != rank:
                    requests.append(self.comm.Isend(src_array[peer:peer + 1], dest=peer))
            MPI.Request.Waitall(requests)
        else:
            # General equal-segment path for larger 1-D buffers.
            for peer in range(nprocs):
                send_start = peer * segment_size
                send_end = (peer + 1) * segment_size
                recv_start = peer * segment_size
                recv_end = (peer + 1) * segment_size

                if peer == rank:
                    # The segment this rank sends to itself does not need MPI communication.
                    dest_array[recv_start:recv_end] = src_array[send_start:send_end]
                else:
                    # Exchange the segment intended for peer with the segment peer
                    # prepared for this rank.
                    self.comm.Sendrecv(
                        src_array[send_start:send_end], dest=peer,
                        recvbuf=dest_array[recv_start:recv_end], source=peer
                    )
        
        # Track bytes transferred by this rank: one send and one receive for
        # every remote rank. The local self-copy is not counted as communication.
        send_seg_bytes = src_array.itemsize * segment_size
        recv_seg_bytes = dest_array.itemsize * segment_size
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)
