! Build module in here, with Block Matrix operations. 
C FILE: kron.f
      Module Kronecker

      contains

      function KronProd(A,B) result(C)
       IMPLICIT NONE
! This should really be a Function!
       real, dimension (:,:), intent(in)  :: A, B
       real, dimension (:,:), allocatable :: C
       integer :: i = 0, j = 0, k = 0, l = 0
       integer :: m = 0, n = 0, p = 0, q = 0


       allocate(C(size(A,1)*size(B,1),size(A,2)*size(B,2)))
       C = 0
  
       do i = 1,size(A,1)
        do j = 1,size(A,2)
         n=(i-1)*size(B,1) + 1
         m=n+size(B,1)
         p=(j-1)*size(B,2) + 1
         q=p+size(B,2) 
         C(n:m,p:q) = A(i,j)*B
        enddo
       enddo
    
      end function KronProd
       
      function DirSum(A,B) result(C)
! This should also be a function!
       real, dimension (:,:), intent(in)  :: A, B
       real, dimension (:,:), allocatable :: C
       integer :: p = 0, q = 0

       allocate(C(size(A,1)+size(B,1),size(A,2)+size(B,2)))
       C = 0

       p = size(A,1) + size(B,1) 
       q = size(A,2) + size(B,2) 

       C(1:size(A,1),1:size(A,2)) = A
       C(size(A,1)+1:p,size(A,2)+1:q) = B

       return
    
       end function DirSum

! Subroutine for the direct sum of two vectors
      function VecDirSum(A,B) result(C)
       real, dimension (:), intent(in)  :: A, B
       real, dimension (:), allocatable :: C

       allocate(C(size(A)+size(B)))
       C = 0

       C(1:size(A)) = A
       C(size(A)+1:size(A)+size(B)) = B
 
       return
    
       end function VecDirSUm
       end module Kronecker
