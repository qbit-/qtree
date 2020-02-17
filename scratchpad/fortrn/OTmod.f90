module OTmod
  !$ use omp_lib
  implicit none

  public :: get_threads

contains

  function get_threads() result(nt)
    integer :: nt

    nt = 0
    !$ nt = omp_get_max_threads()

  end function get_threads

end module OTmod
