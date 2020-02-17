        module alloc
          !$ use omp_lib
          public :: get_threads
          real, allocatable, dimension(:,:) :: a,b,c

        contains
          function get_threads() result(nt)
            integer :: nt

            nt = 0
            !$ nt = omp_get_max_threads()


          end function get_threads

          subroutine foo() 

           integer :: m = 0, n = 0, p = 0, q = 0
           integer ti


            if (allocated(a)) then
               print*, "a=[", size(a,1), size(a,2),"]"
               print*, "b=[", size(b,1), size(b,2),"]"

                !$OMP PARALLEL shared(a,b,c) private(ti,p,q,n,m)

               do i = 1,size(A,1)
                !$OMP DO
                do j = 1,size(A,2)
                    ti = omp_get_thread_num()
                    print*, ti
                 n=(i-1)*size(B,1) + 1
                 m=n+size(B,1)
                 p=(j-1)*size(B,2) + 1
                 q=p+size(B,2) 
                 C(n:m,p:q) = A(i,j)*B
                enddo
                !$OMP end do
               enddo
                !$OMP END PARALLEL
            else
               print*, "a is not allocated"
            endif
          end subroutine foo
        end module alloc
