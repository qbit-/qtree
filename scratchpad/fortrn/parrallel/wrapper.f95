module w_fortranmodule
use fortranmodule, only: test
implicit none

contains
    subroutine w_test(x, y, ysize)
        real(8), intent(in), dimension(ysize)   :: x
        real(8), intent(out), dimension(ysize)  :: y
        integer                                 :: ysize
        !f2py intent(hide) :: ysize=shape(x, 0)
        !f2py depend(ysize) y
        call test(x, y)
    end subroutine w_test
end module w_fortranmodule
