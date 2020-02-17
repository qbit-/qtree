! Build module in here, with Block Matrix operations. 
        module alloc
          real, allocatable, dimension(:,:) :: a,b,c
        contains
          subroutine foo() 
           integer :: i = 0, j = 0, k = 0, l = 0
           integer :: m = 0, n = 0, p = 0, q = 0

            if (allocated(a)) then
               print*, "a=[", size(a,1), size(a,2),"]"
               print*, "]"
               print*, "b=[", size(b,1), size(b,2),"]"
               print*, "]"

               do i = 1,size(A,1)
                do j = 1,size(A,2)
                 n=(i-1)*size(B,1) + 1
                 m=n+size(B,1)
                 p=(j-1)*size(B,2) + 1
                 q=p+size(B,2) 
                 C(n:m,p:q) = A(i,j)*B
                enddo
               enddo
            else
               print*, "a is not allocated"
            endif
          end subroutine foo
        end module alloc
