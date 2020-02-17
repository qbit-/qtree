! Build module in here, with Block Matrix operations. 
C FILE: kron.f
      function KronProd(A,B,ax,bx,ay,by) result(C)
       IMPLICIT NONE
       integer :: ax,ay,bx,by
! This should really be a Function!
       real, dimension (ax, ay), intent(in)  :: a
       real, dimension (bx, by), intent(in)  :: b
Cf2py  integer intent(hide),depend(a)::ax=shape(a,0),by=shape(a,1)
Cf2py  integer intent(hide),depend(b)::bx=shape(b,0),by=shape(b,1)
       real, dimension (ax*bx, ay*by) :: C
       integer :: i = 0, j = 0, k = 0, l = 0
       integer :: m = 0, n = 0, p = 0, q = 0
       print*, size(a,1), size(a,2)
       print*, size(b,1), size(b,2)
       print*, ax,bx, ay,by


  
       do i = 1,size(A,1)
        do j = 1,size(A,2)
         n=(i-1)*size(B,1) + 1
         m=n+size(B,1)
         p=(j-1)*size(B,2) + 1
         q=p+size(B,2) 
         print*, n,p, m,q
         C(n:m,p:q) = A(i,j)*B
        enddo
       enddo
    
      end function KronProd
       
