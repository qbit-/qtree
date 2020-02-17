! fortranmodule.f95 
module fortranmodule
use omp_lib

contains
    subroutine test(x, y)
        real(8), dimension(:), intent(in)  :: x
        real(8), dimension(:), intent(out) :: y
        ! Code begins
        integer :: i, n
        integer :: num_threads
        n = size(x, 1)

        !$omp parallel do private(i) firstprivate(n) shared(x, y)
        do i = 1, n
            if (i == 1) then
                ! The if clause can be removed for serious use.
                ! It is here for debugging only.
                num_threads = OMP_get_num_threads()
                print *, 'num_threads running:', num_threads
            end if
            y(i) = sin(x(i)) + cos(x(i) + exp(x(i))) + log(x(i))
        end do
        !$omp end parallel do
    end subroutine test
end module fortranmodule
