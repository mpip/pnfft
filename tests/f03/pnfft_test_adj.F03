module xgrid_m
  use,intrinsic :: iso_c_binding
  implicit none

contains
  subroutine init_global_cartesian_grid(M_global,lo,up, spacing,x_cartesian,local_M)
    use, intrinsic :: iso_c_binding
    integer(C_INTPTR_T), intent(IN) :: M_global(3)
    real(C_DOUBLE), intent(IN) :: lo(3),up(3)
    real(C_DOUBLE),intent(IN) :: spacing(3)
    real(C_DOUBLE), intent(out) :: x_cartesian(:,:)
    integer(C_INTPTR_T), intent(OUT) :: local_M

    real(C_DOUBLE) :: ext(3)=2.0D0,maxx,minx,center,width
    integer :: idim,ii,cart_local_M

    local_M = 1
    do idim=1,3 
       do ii = 1, M_global(idim) 
          x_cartesian(ii, idim) = (ii - int(M_global(idim)/2) -1) * spacing(idim)
       end do

       if (ext(idim) > 1.0D0 ) then
          x_cartesian(1,              idim) = (-int(M_global(idim)/2)) * spacing(idim) * ext(idim)
          x_cartesian(M_global(idim), idim) = (int(M_global(idim)/2))  * spacing(idim) * ext(idim)
       end if

       ! Now rescale the coordinates to [-0.5,0.5)
       maxx=maxval(x_cartesian(1:M_global(idim),idim))
       minx=minval(x_cartesian(1:M_global(idim),idim))
       width=maxx-minx
       center=(maxx+minx)/2.0D0

       do ii=1,M_global(idim)
          x_cartesian(ii,idim) = (x_cartesian(ii,idim)-center)/(0.50D0*width/0.4D0)
       end do

       cart_local_M = 0
       do ii=1,M_global(idim)
         if ( (x_cartesian(ii,idim).ge.lo(idim)) .and. (x_cartesian(ii,idim).lt.up(idim)) ) then
           cart_local_M = cart_local_M + 1
         end if
       end do
       local_M = local_M*cart_local_M
    end do
    !print*,"local_M = ",local_M
  end subroutine init_global_cartesian_grid

  subroutine init_local_grid(lo,up,M_global,xcart,x)
    real(C_DOUBLE),intent(IN) :: lo(3),up(3)
    integer(C_INTPTR_T) :: M_global(3)
    real(C_DOUBLE),intent(IN) :: xcart(:,:)
    real(C_DOUBLE),intent(OUT) :: x(:,:)

    integer :: iPoint,i,j,k
    ! all given arrays are in Fortran order

    !print*,"lo = ",lo
    !print*,"up = ",up
    iPoint=1
    do k=1,M_global(3)
      !write(*,"(A,I3,A,ES20.12)",advance='no') "xcart(",k,",3) = ",xcart(k,3)
       if ( (xcart(k,3).ge.lo(3)).and.(xcart(k,3).lt.up(3)) ) then
         !write(*,"(A)") "...yes"
         do j=1,M_global(2)
           if ( (xcart(j,2).ge.lo(2)).and.(xcart(j,2).lt.up(2)) ) then
             do i=1,M_global(1)
               if ( (xcart(i,1).ge.lo(1)).and.(xcart(i,1).lt.up(1)) ) then
                 x(1,iPoint)=xcart(k,3)
                 x(2,iPoint)=xcart(j,2)
                 x(3,iPoint)=xcart(i,1)
                 iPoint = iPoint + 1
               end if
             end do
           end if
         end do
       !else
       !  write(*,"(A)") "...no"
       end if
     end do
     iPoint = iPoint-1
     !print*,"iPoint = ",iPoint
  end subroutine init_local_grid
end module xgrid_m

module debug_output_m
  use,intrinsic :: iso_c_binding
  use mpi
  implicit none
  interface control_sum
    module procedure control_sum_cmplx_1d!, control_sum_real_1d
  end interface control_sum
contains
  
  function control_sum_cmplx_1d(arr,comm) result(total)
    complex(C_DOUBLE_COMPLEX) :: arr(:)
    real(C_DOUBLE) :: total
    integer, optional :: comm

    real(C_DOUBLE) :: local
    integer :: comm_, mpi_err

    if (present(comm)) then
      comm_=comm
    else
      comm_=-1
    end if
    local = sum(real(arr*conjg(arr)))
    if (comm_ .ne. -1 ) then
      total = 0.0D0
      call MPI_Reduce(local,total,1,MPI_DOUBLE,MPI_SUM,0,comm_,mpi_err)
    else
      total=local
    end if
  end function control_sum_cmplx_1d

end module debug_output_m

program main
  use, intrinsic :: iso_c_binding
  use mpi
  use xgrid_m
  use debug_output_m
  implicit none
  !include "mpif.h"
  include "fftw3-mpi.f03"
  include "pfft.f03"
  include "pnfft.f03"

  integer :: np(3), np_for_c(3), ierror, l1, l2, l3
  integer(C_INTPTR_T) :: N(3), local_N(3), local_N_start(3),M_global(3)
  integer(C_INTPTR_T) :: Nos(3), Nos_for_c(3)
  integer(C_INTPTR_T),dimension(3) :: N_for_c, local_N_for_c, local_N_start_for_c
  integer(C_INTPTR_T) :: local_M
  integer :: in_local_M
  integer(C_INT),parameter :: d=3
  real(C_DOUBLE) :: lower_border(3), upper_border(3)
  real(C_DOUBLE) :: lower_border_for_c(3), upper_border_for_c(3)
  type(C_PTR) :: pnfft, cf_hat, cf, cx
  complex(C_DOUBLE_COMPLEX), pointer :: f_hat(:,:,:), f(:)
  real(C_DOUBLE), pointer :: x(:,:)
  real(C_DOUBLE),allocatable, dimension(:,:) :: cartesian_grid
  real(C_DOUBLE) :: x_max(3),x_max_for_c(3)
  integer(C_INT) :: mm

  integer :: comm_cart_3d, myrank, dims(3),dims_for_c(3),coords(3),n_procs
  logical :: periods(3), use_f_from_octopus=.false.
  integer :: fh,i,j,k,ii
  character(len=100) :: filename
  integer :: pnfft_flags,fftw_flags
  real(C_DOUBLE) :: radius,spacing(3),global_sum
  real(C_DOUBLE) :: sigma

  ! Initialize MPI
  call MPI_Init(ierror)
  call MPI_Comm_size(MPI_COMM_WORLD, n_procs, ierror)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierror)

  ! try to find a suitable parallel distribution
  call compute_parallel_distribution(np)
  print*,"np(Fortran) = ",np

  ! -----------------------------
  ! Input parameters set here
  radius=10.0D0
  !spacing= (/2.0D0,2.8D0,1.4D0/)
  spacing= (/1.2D0,1.1D0,1.4D0/)
  sigma = 1.1D0
  mm=6
  ! -----------------------------


  ! determine the number of grid points from the radius and the spacing
  call get_N(radius,spacing,N)
  !print*,"N = ",N
  !N = (/ 6,8,10 /)
  !Nos = (/ 12,16,24 /)

!! ----- the following line does lead to a segfault if run with 3 (three) processes
  !Nos = (/ 12,10,16 /)
!! ----- it works with the following line
  !Nos = (/ 12,10,18 /)
   
  call pnfft_init();

  ! Create three-dimensional process grid of
  ! size np(1) x np(2) x np(3), if possible
  np_for_c=np(3:1:-1)
  ierror =  pnfft_create_procmesh(3, MPI_COMM_WORLD, np_for_c, comm_cart_3d)
  if (ierror .ne. 0) then
     if(myrank .eq. 0) then
        write(*,*) "Error: This test file only works with ", np(1)*np(2)*np(3), " processes"
     endif
     call MPI_Finalize(ierror)
     call exit(1)
  endif

  call MPI_Cart_get(comm_cart_3d,3,dims_for_c,periods,coords,ierror)
  dims = dims_for_c(3:1:-1)

  ! determine Nos to be a multiple of the number of PEs in the processor grid
  Nos = 0
  do ii = 1, 3
    Nos(ii) = N(ii)*sigma
    
    ! We also must make sure that Nos(ii) is divisable by dims(ii)
    do while ((modulo(Nos(ii),int(dims(ii),C_INTPTR_T)).ne.0).or.(modulo(Nos(ii),2_8).ne.0))
      Nos(ii) = Nos(ii) + 1
    end do
  end do

  ! for testing purposes make sure that Nos is equal for 2 and 3 PEs so that the two
  ! parallelizations can be compared. It is parallelized only in the third dimension.
  ii = 3
  do while ((modulo(Nos(ii),int(dims(ii),C_INTPTR_T)).ne.0).or.(modulo(Nos(ii),2_8).ne.0).or.(modulo(Nos(ii),3_8).ne.0))
    Nos(ii) = Nos(ii) + 1
  end do

  ! set and get the local size for the pnfft. Be aware of different order in Fortran
  ! and C
  N_for_c=N(3:1:-1)
  call pnfft_local_size_3d(N_for_c, comm_cart_3d, PNFFT_TRANSPOSED_NONE, &
       local_N_for_c, local_N_start_for_c, lower_border_for_c, upper_border_for_c)

  local_N=local_N_for_c(3:1:-1)
  local_N_start=local_N_start_for_c(3:1:-1)
  lower_border=lower_border_for_c(3:1:-1)
  upper_border=upper_border_for_c(3:1:-1)
  !write(*,"(A,3I5)") "local_N (Fortran) =",local_N
  !write(*,"(A,3I5)") "local_N_start (Fortran) =",local_N_start
  !write(*,"(A,3ES11.3)") "lower_border (Fortran) = ",lower_border
  !write(*,"(A,3ES11.3)") "upper_border (Fortran) = ",upper_border

  ! Initialize global Cartesian grid
  ! the same number of gridpoints as we have as Fourier modes
  M_global=N
  allocate(cartesian_grid(maxval(M_global),3))
  cartesian_grid=0.0D0
  call init_global_cartesian_grid(M_global,lower_border,upper_border,spacing,cartesian_grid,local_M)
  !print*,"local_M = ",local_M
  print*,"M_global = ",M_global
  do ii=1,3
    x_max(ii)=maxval(abs(cartesian_grid(1:M_global(ii),ii)))
  end do
  !x_max = 0.5D0
  ! Plan parallel NFFT
  !pnfft = pnfft_init_3d(N_for_c, local_M, comm_cart_3d)
  pnfft_flags = PNFFT_MALLOC_X + PNFFT_MALLOC_F_HAT + PNFFT_MALLOC_F + &
    &           PNFFT_WINDOW_KAISER_BESSEL
  fftw_flags = PFFT_ESTIMATE
  !pnfft = pnfft_init_adv(d, N_for_c, local_M, pnfft_flags,fftw_flags, comm_cart_3d)
  Nos_for_c = Nos(3:1:-1)
  x_max_for_c = x_max(3:1:-1)
  write(*,"(A,3I4)") "N_for_c   = ", N_for_c
  write(*,"(A,3I4)") "Nos_for_c = ", Nos_for_c
  write(*,"(A,3ES11.3)")  "x_max_for_c = ", x_max_for_c
  print*,"real space cutoff mm = ",mm
  pnfft = pnfft_init_guru(d, N_for_c, Nos_for_c, &
    &                     x_max_for_c, local_M, mm, &
    &                     pnfft_flags, fftw_flags, comm_cart_3d)


  ! Get data pointers in C format
  cf_hat = pnfft_get_f_hat(pnfft)
  cf     = pnfft_get_f(pnfft)
  cx     = pnfft_get_x(pnfft)

  ! Convert data pointers to Fortran format
  call c_f_pointer(cf_hat, f_hat, [local_N])
  call c_f_pointer(cf,     f,     [local_M])
  call c_f_pointer(cx,     x,     [int(d,C_INTPTR_T),local_M])

  call init_local_grid(lower_border,upper_border,M_global,cartesian_grid,x)

  ! Initialize Fourier coefficients
  call init_f_hat(N, local_N, local_N_start, &
    &             f_hat)

  ! Output for debugging
  write(filename,"(A,I2.2,A,I2.2,A)") "xgrid_np",n_procs,"_",myrank,".dat"
  open(newunit=fh,file=trim(filename))
  do i=1,local_M
    write(fh,"(3ES20.12)") x(:,i)
  end do
  close(fh)
  write(filename,"(A,I2.2,A,I2.2,A)") "fhat_np",n_procs,"_",myrank,".dat"
  open(newunit=fh,file=trim(filename))
  do k=1,local_N(3)
    do j=1,local_N(2)
      do i=1,local_N(1)
        write(fh,"(2ES20.12)") f_hat(i,j,k)
      end do
    end do
  end do
  close(fh)

  ! Execute parallel NFFT
  call pnfft_trafo(pnfft)

  write(filename,"(A,I2.2,A,I2.2,A)") "xf_np",n_procs,"_",myrank,".dat"
  open(newunit=fh,file=trim(filename))
  do i=1,local_M
    write(fh,"(5ES20.12)") x(:,i),f(i)
  end do
  close(fh)

  if (use_f_from_octopus) then
    ! read in the f values from octopus from a binary file
    ! this is necessary to have always the same input, even if mm or sigma is changed
    ! Both parameters would also influence the pnfft_trafo call and would therefore
    ! give different inputs, which I want to avoid for getting the influence of
    ! these parameters solely on the pnfft_adj routine.
    write(filename,"(2(A,I2.2),A)") "fvals_adj_np",n_procs,"_",myrank,"_00000.bin"
    open(newunit=fh,file=trim(filename),form='unformatted')
    read(fh) in_local_M
    if (in_local_M.ne.int(local_M,C_INT)) then
      print*,"local_M of ",filename," is different from local local_M."
      close(fh)
      stop
    end if
    read(fh) f
    close(fh)
  end if

  ! compute test sum of f
  global_sum = control_sum(f,comm_cart_3d)
  if (myrank.eq.0) print*,"global_sum of f from file = ",global_sum

  ! Execute parallel adjoint NFFT
  call pnfft_adj(pnfft)

  ! Scale data
  do l3=1,local_N(3)
     do l2=1,local_N(2)
        do l1=1,local_N(1)
           f_hat(l1,l2,l3) = f_hat(l1,l2,l3) / (N(3)*N(2)*N(1))
        enddo
     enddo
  enddo

  ! Print output Fourier coefficents
  write(filename,"(A,I2.2,A,I2.2,A)") "fhat2_np",n_procs,"_",myrank,".dat"
  open(newunit=fh,file=trim(filename))
  do k=1,local_N(3)
    do j=1,local_N(2)
      do i=1,local_N(1)
        write(fh,"(2ES20.12)") f_hat(i,j,k)
      end do
    end do
  end do
  close(fh)

  call MPI_Barrier(comm_cart_3d,ierror)

  ! Free mem and finalize      
  call pnfft_finalize(pnfft, &
    &                 PNFFT_FREE_X + PNFFT_FREE_F_HAT + PNFFT_FREE_F)
  call MPI_Comm_free(comm_cart_3d, ierror)
  call pnfft_cleanup()
  call MPI_Finalize(ierror)

contains
  subroutine compute_parallel_distribution(np)
    integer, dimension(3) :: np

    integer,parameter :: n_primes = 9
    integer :: primes(n_primes)=[2,3,5,7,11,13,17,19,23]
    integer :: actDim,rem,iPrime
    integer :: n_procs,ierror

    call MPI_Comm_size(MPI_COMM_WORLD, n_procs, ierror)

    np = 1
    actDim = 3
    rem = n_procs
    do iPrime=1,n_primes
      do while (modulo(rem,primes(iPrime)) == 0)
        np(actDim) = np(actDim) * primes(iPrime)
        rem = rem/primes(iPrime)
        actDim = modulo(actDim-2+3,3)+1
      end do
    end do
  end subroutine compute_parallel_distribution

  subroutine get_N(radius,spacing,Nout)
    real(C_DOUBLE) :: radius
    real(C_DOUBLE),intent(IN) :: spacing(3)
    integer(C_INTPTR_T),intent(OUT) :: Nout(3)

    integer :: nr(2,3)
    integer :: idir,jj
    logical :: out
    real(C_DOUBLE) :: chi
    nr = 0
    do idir = 1, 3
      chi = 0.0D0
      ! the upper border
      jj = 0
      out = .false.
      do while(.not.out)
        jj = jj + 1
        chi = real(jj, C_DOUBLE)*spacing(idir)
        out = (chi > nearest(radius, 1.0D0))
      end do
      nr(2, idir) = jj - 1
    end do

    ! we have a symmetric mesh (for now)
    nr(1,:) = -nr(2,:)
    Nout = nr(2,:)-nr(1,:)+1
    ! due to the two-point enlargement, we have to add 2 grid points
    Nout = Nout + 2

    ! and now we have to make sure that the number is even
    do idir=1,3
      if (modulo(Nout(idir),2_8).ne.0) Nout(idir) = Nout(idir)+1
    end do
  end subroutine get_N
end program main

subroutine init_f_hat(N, local_N, local_N_start, f_hat)
  use, intrinsic :: iso_c_binding
  implicit none

  integer(C_INTPTR_T) :: l1, l2, l3
  integer(C_INTPTR_T), intent(in) :: N(3), local_N(3), local_N_start(3)
  complex(C_DOUBLE_COMPLEX), intent(out) :: f_hat(local_N(1),local_N(2),local_N(3))

  ! use C-like row-major order here  
  do l1 = 1,local_N(1)
     do l2 = 1,local_N(2)
        do l3 = 1,local_N(3)
           f_hat(l1,l2,l3) = &
                func(l1-1 + local_N_start(1), N(1)) & 
                * func(l2-1 + local_N_start(2), N(2)) &
                * func(l3-1 + local_N_start(3), N(3)) 
        enddo
     enddo
  enddo

contains
  real function func(l,N)
    use, intrinsic :: iso_c_binding
    implicit none
    
    integer(C_INTPTR_T), intent(in) :: l, N
    real(C_DOUBLE) :: amplitude
    
    amplitude=2.0D-3
    func = amplitude * exp(-real(l*l)/real(N*N))
  end function func
  
end subroutine init_f_hat

