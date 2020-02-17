module OTmod
  !$ use omp_lib

  public :: get_threads
  real, allocatable, dimension(:,:) :: a,b,c

contains
  function pra() result(x)
      integer ti
      integer :: m = 0, n = 0, p = 0, q = 0
    !$OMP PARALLEL shared(a,b,c) private(ti,p,q,n,m)

    ti = omp_get_thread_num()
    print*, 'thr',ti
    do i = 1, size(a,1)
        !$OMP DO
        do j = 1, size(a,2)
             n=(i-1)*size(B,1) + 1
             m=n+size(B,1)
             p=(j-1)*size(B,2) + 1
             q=p+size(B,2) 
             C(n:m,p:q) = A(i,j)*B
        enddo
        !$OMP end do
    enddo

    !$OMP END PARALLEL
    x=0
  end function pra

  function get_threads() result(nt)
    integer :: nt

    nt = 0
    !$ nt = omp_get_max_threads()


  end function get_threads

  function get_thread_id() result(ti)
    integer :: ti, i
    !$OMP PARALLEL PRIVATE(ti)

    ti = omp_get_thread_num()
    DO i=1,OMP_GET_MAX_THREADS()
        IF (i == ti) THEN
            PRINT *, "Hello from process: ", ti
        END IF
        !$OMP BARRIER
    END DO
    !$OMP END PARALLEL
    ti = 0
    !$ ti = omp_get_thread_num()


  end function get_thread_id

end module OTmod
