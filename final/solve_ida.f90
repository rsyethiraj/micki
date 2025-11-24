module solve_ida

   implicit none

   integer :: neq = 6
   integer :: iout(50)
   real*8 :: rout(50)
   real*8 :: y0(6), yp0(6)
   real*8 :: diff(6), mas(6, 6)
   real*8 :: jac(6, 6)
   real*8 :: rates(11)
   real*8 :: dypdr(6, 11)
   integer :: dvacdy(1, 6)

end module solve_ida

subroutine initialize(neqin, y0in, rtol, atol, ipar, rpar, id_vec)

   use solve_ida, only: neq, iout, rout, y0, yp0, mas, diff, dypdr, dvacdy

   implicit none

   integer, intent(in) :: neqin, ipar(*)
   real*8, intent(in) :: y0in(neqin), rtol, atol(*)
   real*8, intent(in) :: rpar(*)
   real*8, intent(in) :: id_vec(neqin)
   real*8 :: constr_vec(neqin)
   real*8 :: t0, yptmp(neqin)
   integer :: nthreads, iatol, ier
   integer :: i
   integer :: meth, itmeth
   integer, external :: OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
   integer :: myid

   dypdr = 0
   dypdr(1, 1) = -1.00000000000000d0
   dypdr(1, 2) = 1.00000000000000d0
   dypdr(1, 4) = 1.00000000000000d0
   dypdr(1, 5) = 1.00000000000000d0
   dypdr(1, 8) = -1.00000000000000d0
   dypdr(1, 11) = -2.00000000000000d0
   dypdr(2, 1) = 1.00000000000000d0
   dypdr(2, 5) = -1.00000000000000d0
   dypdr(2, 6) = -1.00000000000000d0
   dypdr(2, 11) = 1.00000000000000d0
   dypdr(3, 1) = 1.00000000000000d0
   dypdr(3, 2) = 1.00000000000000d0
   dypdr(3, 7) = 1.00000000000000d0
   dypdr(3, 9) = 2.00000000000000d0
   dypdr(4, 2) = -1.00000000000000d0
   dypdr(4, 8) = 1.00000000000000d0
   dypdr(4, 10) = 1.00000000000000d0
   dypdr(4, 11) = 1.00000000000000d0
   dypdr(5, 3) = 1.00000000000000d0
   dypdr(5, 4) = 1.00000000000000d0
   dypdr(5, 6) = -1.00000000000000d0
   dypdr(6, 4) = -1.00000000000000d0
   dypdr(6, 5) = -1.00000000000000d0
   dypdr(6, 7) = -1.00000000000000d0
   dypdr(6, 8) = -1.00000000000000d0

   dvacdy = 0
   dvacdy(1, 1) = -1
   dvacdy(1, 2) = -1
   dvacdy(1, 3) = -1
   dvacdy(1, 4) = -1
   dvacdy(1, 5) = -1
   dvacdy(1, 6) = -1

   iatol = 2
   constr_vec = 1.d0

   !$OMP PARALLEL DEFAULT(private) SHARED(nthreads)
   myid = OMP_GET_THREAD_NUM()
   if (myid == 0) then
      nthreads = OMP_GET_NUM_THREADS()
   end if
   !$OMP END PARALLEL

   y0 = y0in
   yp0 = 0
   yptmp = 0
   diff = id_vec
   mas = 0
   t0 = 0
   meth = 2  ! 1 = Adams (nonstiff), 2 = BDF (stiff)
   itmeth = 2  ! 1 = functional iteration, 2 = Newton iteration

   do i = 1, neq
      mas(i, i) = id_vec(i)
   enddo

!   ! Calculate yp
   call fidaresfun(0.d0, y0, yptmp, yp0, ipar, rpar, ier)

   ! initialize Sundials
   call fnvinits(2, neq, ier)
   !call fnvinitomp(2, neq, nthreads, ier)
   ! allocate memory
   call fidamalloc(t0, y0, yp0, iatol, rtol, atol, iout, rout, ipar, rpar, ier)
! TODO: Fix all of these settings
   ! set maximum number of steps (default = 500)
   call fidasetiin('MAX_NSTEPS', 50000, ier)
   ! set algebraic variables
   call fidasetvin('ID_VEC', id_vec, ier)
   ! set constraints (all yi >= 0.)
   call fidasetvin('CONSTR_VEC', constr_vec, ier)
!   ! enable stability limit detection
!   call fidasetiin('STAB_LIM', 1, ier)
   ! initialize the solver
   call fidalapackdense(neq, ier)
   ! enable the jacobian
   call fidalapackdensesetjac(1, ier)
!   ! initialize the solver
!   call fidaspgmr(0, 0, 10, 0.05d0, 1.0d0, ier)
!!   call fidaspbcg(0, 0.05d0, 1.0d0, ier)
!!   call fidasptfqmr(0, 0.05d0, 1.0d0, ier)
!   ! enable the jacobian
!   call fidaspilssetjac(1, ier)
!   ! enable the preconditioner
!   call fidaspilssetprec(1, ier)

end subroutine initialize

subroutine find_steady_state(neqin, nrates, dt, maxiter, epsilon, t1, u1, du1, r1)

   use solve_ida, only: y0, yp0, iout, rout, rates, dypdr

   implicit none

   integer, intent(in) :: neqin, nrates, maxiter
   real*8, intent(in) :: dt, epsilon

   real*8 :: rpar(1)
   integer :: ipar(1)

   real*8, intent(out) :: t1, u1(neqin), du1(neqin), r1(nrates)

   real*8 :: tout, epsilon2
   real*8 :: dutmp(neqin), du0(neqin)
   integer :: itask, ier
   integer :: i

   logical :: converged = .FALSE.

   epsilon2 = epsilon**2
   i = 0
   itask = 1
   tout = 0.0d0
   u1 = y0
   du1 = yp0
   t1 = 0.d0
   du0 = 0.d0

   call fidacalcic(1, dt, ier)

   do while (.not. converged)
      if (tout - t1 < dt * 0.01) then
         tout = tout + dt
      end if

      call fidasolve(tout, t1, u1, du1, itask, ier)

      i = i + 1

      call fidaresfun(tout, u1, du0, dutmp, ipar, rpar, ier)

      if (maxval(dutmp**2) < epsilon2) then
         converged = .TRUE.
      end if
      if (i >= maxiter) then
         print *, "ODE NOT CONVERGED!"
         exit
      end if
   end do
   
   call ratecalc(6, u1)
   r1 = rates

end subroutine find_steady_state

subroutine solve(neqin, nrates, nt, tfinal, t1, u1, du1, r1)

   use solve_ida, only: y0, yp0, iout, rout, rates

   implicit none

   integer, intent(in) :: neqin, nt, nrates
   real*8, intent(in) :: tfinal

   real*8 :: rpar(1)
   integer :: ipar(1)

   real*8, intent(out) :: t1(nt)
   real*8, intent(out) :: u1(neqin, nt), du1(neqin, nt)
   real*8, intent(out) :: r1(nrates, nt)

   real*8 :: dt, tout
   integer :: itask, ier
   integer :: i

   itask = 1
   dt = tfinal / (nt - 1)
   tout = 0.0d0
   u1 = 0
   du1 = 0
   u1(:, 1) = y0
   du1(:, 1) = yp0
   t1(1) = 0.d0
   call ratecalc(6, u1(:, 1))
   r1(:, 1) = rates


!   call fidacalcic(1, dt, ier)

   do i = 2, nt
      tout = tout + dt
      do while (tout - t1(i) > dt * 0.01)
         call fidasolve(tout, t1(i), u1(:, i), du1(:, i), itask, ier)
      end do
      r1(:, i) = rates
   end do

end subroutine solve

subroutine finalize

   implicit none

   call fidafree

end subroutine finalize

subroutine fidaresfun(tres, yin, ypin, res, ipar, rpar, reserr)

   use solve_ida, only: neq, diff, dypdr, rates

   implicit none

   integer, intent(in) :: ipar(*)
   integer, intent(out) :: reserr
   real*8, intent(in) :: tres, rpar(*)
   real*8, intent(in) :: yin(neq), ypin(neq)
   real*8, intent(out) :: res(neq)
   real*8 :: y(neq)

   integer :: i

   reserr = 0

   y = yin
   res = 0


   do i = 1, neq
      if (y(i) < -1d-10)  then
!         y(i) = 0.d0
         reserr = 1
      endif
   enddo

   call ratecalc(6, y)

   res = matmul(dypdr, rates) - diff * ypin
   
end subroutine fidaresfun

subroutine fidadjac(neqin, t, yin, ypin, r, jac, cj, ewt, h, ipar, rpar, wk1, wk2, wk3, djacerr)

   use solve_ida, only: mas, dypdr, dvacdy
    
   implicit none
   
   integer :: neqin, ipar(*)
   integer :: djacerr
   real*8 :: t, h, cj, rpar(*)
   real*8 :: yin(neqin), ypin(neqin), r(neqin), ewt(*), jac(neqin, neqin)
   real*8 :: wk1(*), wk2(*), wk3(*)
   real*8 :: y(neqin), drdy(11, 6), drdvac(11, 1), vac(1)

   integer :: i

   djacerr = 0

   jac = 0
   y = yin

   do i = 1, neqin
      if (y(i) < -1d-10) then
         y(i) = 0.d0
         djacerr = 1
      endif
   enddo

   vac = 0
   vac(1) = -y(5) - y(6) - y(3) - y(4) - y(2) - y(1) + 1

   do i = 1, 1
      if (vac(i) < -1d-10) then
         vac(i) = 0.d0
         djacerr = 1
      endif
   enddo

   drdy = 0
   drdy(1, 1) = 63.8798753446516d0*vac(1)*exp(-15.4081163912808d0*y(5) - 30.7473240605402d0* &
      y(2))
   drdy(1, 2) = -25358535.026478d0*y(3)*y(2)*exp(-15.4081163912808d0*y(5) - 30.7473240605402d0*y(2) &
      )*exp(24.8873719748563d0*y(5) + 49.6634417662587d0*y(2)) - &
      1340578.20008235d0*y(3)*exp(-15.4081163912808d0*y(5) - &
      30.7473240605402d0*y(2))*exp(24.8873719748563d0*y(5) + &
      49.6634417662587d0*y(2)) - 1964.13522816891d0*y(1)*vac(1)*exp( &
      -15.4081163912808d0*y(5) - 30.7473240605402d0*y(2))
   drdy(1, 3) = -1340578.20008235d0*y(2)*exp(-15.4081163912808d0*y(5) - 30.7473240605402d0*y(2)) &
      *exp(24.8873719748563d0*y(5) + 49.6634417662587d0*y(2))
   drdy(1, 5) = -12707683.3883501d0*y(3)*y(2)*exp(-15.4081163912808d0*y(5) - 30.7473240605402d0* &
      y(2))*exp(24.8873719748563d0*y(5) + 49.6634417662587d0*y(2)) - &
      984.268554370903d0*y(1)*vac(1)*exp(-15.4081163912808d0*y(5) - &
      30.7473240605402d0*y(2))
   drdy(2, 1) = -66299103748.64d0*y(3)*exp(1.97619170776594d0*y(5))
   drdy(2, 3) = -66299103748.64d0*y(1)*exp(1.97619170776594d0*y(5))
   drdy(2, 4) = 261721.165850268d0*vac(1)*exp(-6.79547637531067d0*y(5))
   drdy(2, 5) = -131019739060.376d0*y(3)*y(1)*exp(1.97619170776594d0*y(5)) - 1778519.99945426d0 &
      *y(4)*vac(1)*exp(-6.79547637531067d0*y(5))
   drdy(3, 1) = -13.474815711096d0*y(5)*exp(34.8099614189798d0*y(5) + 42.1998123408802d0* &
      y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1))
   drdy(3, 2) = -58.7348583071033d0*y(5)*exp(34.8099614189798d0*y(5) + 42.1998123408802d0* &
      y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1))
   drdy(3, 3) = -12.172595228728d0*y(5)*exp(34.8099614189798d0*y(5) + 42.1998123408802d0* &
      y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1))
   drdy(3, 4) = -5.40501505153578d0*y(5)*exp(34.8099614189798d0*y(5) + 42.1998123408802d0* &
      y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1))
   drdy(3, 5) = -80.3310172278976d0*y(5)*exp(34.8099614189798d0*y(5) + 42.1998123408802d0* &
      y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1)) - 2.30770198969821d0 &
      *exp(34.8099614189798d0*y(5) + 42.1998123408802d0*y(6) + &
      5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + 25.4516651497035d0 &
      *y(2) + 5.8390623101461d0*y(1))
   drdy(3, 6) = -97.3845909039403d0*y(5)*exp(34.8099614189798d0*y(5) + 42.1998123408802d0* &
      y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1))
   drdy(4, 1) = -3419348570.9252d0*y(5)*y(1)*exp(-1.55078861175432d0*y(5) + 42.1998123408802d0 &
      *y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1))*exp( &
      0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) - &
      1594735764.81432d0*y(5)*exp(-1.55078861175432d0*y(5) + &
      42.1998123408802d0*y(6) + 5.2747691352989d0*y(3) + &
      2.34216336236839d0*y(4) + 25.4516651497035d0*y(2) + 5.8390623101461d0 &
      *y(1))*exp(0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) - &
      9928070.58050501d0*y(6)*vac(1)*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1))
   drdy(4, 2) = -14904467572.8298d0*y(5)*y(1)*exp(-1.55078861175432d0*y(5) + &
      42.1998123408802d0*y(6) + 5.2747691352989d0*y(3) + &
      2.34216336236839d0*y(4) + 25.4516651497035d0*y(2) + 5.8390623101461d0 &
      *y(1))*exp(0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) - &
      43275086.7478435d0*y(6)*vac(1)*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1))
   drdy(4, 3) = -3088899098.30289d0*y(5)*y(1)*exp(-1.55078861175432d0*y(5) + &
      42.1998123408802d0*y(6) + 5.2747691352989d0*y(3) + &
      2.34216336236839d0*y(4) + 25.4516651497035d0*y(2) + 5.8390623101461d0 &
      *y(1))*exp(0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) - &
      8968611.30940844d0*y(6)*vac(1)*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1))
   drdy(4, 4) = -1371568330.76976d0*y(5)*y(1)*exp(-1.55078861175432d0*y(5) + &
      42.1998123408802d0*y(6) + 5.2747691352989d0*y(3) + &
      2.34216336236839d0*y(4) + 25.4516651497035d0*y(2) + 5.8390623101461d0 &
      *y(1))*exp(0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) - &
      3982345.44136668d0*y(6)*vac(1)*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1))
   drdy(4, 5) = 908140133.081832d0*y(5)*y(1)*exp(-1.55078861175432d0*y(5) + 42.1998123408802d0 &
      *y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1))*exp( &
      0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) + &
      2636782.75297512d0*y(6)*vac(1)*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1)) - 1594735764.81432d0*y(1)*exp( &
      -1.55078861175432d0*y(5) + 42.1998123408802d0*y(6) + &
      5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + 25.4516651497035d0 &
      *y(2) + 5.8390623101461d0*y(1))*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1))
   drdy(4, 6) = -24712164446.3989d0*y(5)*y(1)*exp(-1.55078861175432d0*y(5) + &
      42.1998123408802d0*y(6) + 5.2747691352989d0*y(3) + &
      2.34216336236839d0*y(4) + 25.4516651497035d0*y(2) + 5.8390623101461d0 &
      *y(1))*exp(0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) - &
      71751711.6877354d0*y(6)*vac(1)*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1)) + 2686955.14446704d0*vac(1)*exp( &
      0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1))
   drdy(5, 2) = 10897614040166.0d0*y(6)
   drdy(5, 6) = 10897614040166.0d0*y(2)
   drdy(6, 1) = 364722.279258857d0*y(5)*y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0* &
      y(6) + 1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1))
   drdy(6, 2) = 4691875.4323889d0*y(5)*y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0* &
      y(6) + 1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1)) + 206758.569691054d0 &
      *y(5)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0*y(6) + &
      1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1))
   drdy(6, 3) = 329475.131349018d0*y(5)*y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0* &
      y(6) + 1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1))
   drdy(6, 4) = 146297.318738188d0*y(5)*y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0* &
      y(6) + 1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1))
   drdy(6, 5) = 3764090.29850314d0*y(5)*y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0* &
      y(6) + 1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1)) + 206758.569691054d0 &
      *y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0*y(6) + &
      1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1))
   drdy(6, 6) = 2635904.69218281d0*y(5)*y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0* &
      y(6) + 1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1))
   drdy(7, 5) = 812435641.955577d0*y(6)*vac(1)*exp(14.3581999433635d0*y(5))
   drdy(7, 6) = 56583391.0351063d0*vac(1)*exp(14.3581999433635d0*y(5))
   drdy(8, 1) = 10897614040166.0d0*y(6)
   drdy(8, 6) = 10897614040166.0d0*y(1)
   drdy(9, 3) = -306385665.697254d0*y(3)*exp(10.5495382705978d0*y(5))
   drdy(9, 5) = -1616113652.91788d0*y(3)**2*exp(10.5495382705978d0*y(5))
   drdy(10, 4) = -253630218456.946d0*exp(2.34216336236839d0*y(5))
   drdy(10, 5) = -594043405259.351d0*y(4)*exp(2.34216336236839d0*y(5))
   drdy(11, 1) = 2.86703872568524d+27*y(1)*exp(-16.1157038917797d0*y(5) - 49.6634417662587d0* &
      y(2))/(131544332324398.0d0*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2)) + 10897614040166.0d0)
   drdy(11, 2) = -7.75839343986673d+41*y(4)*y(2)*exp(-32.2314077835593d0*y(5) - &
      99.3268835325174d0*y(2))*exp(16.1157038917797d0*y(5) + &
      49.6634417662587d0*y(2))/(131544332324398.0d0*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2)) + &
      10897614040166.0d0)**2 - 1.18757991768423d+26*y(4)*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2))*exp( &
      16.1157038917797d0*y(5) + 49.6634417662587d0*y(2))/( &
      131544332324398.0d0*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2)) + 10897614040166.0d0) - &
      7.11935053973387d+28*y(1)**2*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2))/(131544332324398.0d0*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2)) + &
      10897614040166.0d0) + 9.36510213332634d+42*y(1)**2*exp( &
      -32.2314077835593d0*y(5) - 99.3268835325174d0*y(2))/( &
      131544332324398.0d0*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2)) + 10897614040166.0d0)**2
   drdy(11, 4) = -1.18757991768423d+26*y(2)*exp(-16.1157038917797d0*y(5) - 49.6634417662587d0* &
      y(2))*exp(16.1157038917797d0*y(5) + 49.6634417662587d0*y(2))/( &
      131544332324398.0d0*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2)) + 10897614040166.0d0)
   drdy(11, 5) = -2.5175857110605d+41*y(4)*y(2)*exp(-32.2314077835593d0*y(5) - &
      99.3268835325174d0*y(2))*exp(16.1157038917797d0*y(5) + &
      49.6634417662587d0*y(2))/(131544332324398.0d0*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2)) + &
      10897614040166.0d0)**2 - 2.31021735747043d+28*y(1)**2*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2))/( &
      131544332324398.0d0*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2)) + 10897614040166.0d0) + &
      3.03895999812683d+42*y(1)**2*exp(-32.2314077835593d0*y(5) - &
      99.3268835325174d0*y(2))/(131544332324398.0d0*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2)) + &
      10897614040166.0d0)**2

   drdvac = 0
   drdvac(1, 1) = 63.8798753446516d0*y(1)*exp(-15.4081163912808d0*y(5) - 30.7473240605402d0*y(2))
   drdvac(2, 1) = 261721.165850268d0*y(4)*exp(-6.79547637531067d0*y(5))
   drdvac(3, 1) = 22338886.1940868d0
   drdvac(4, 1) = 2686955.14446704d0*y(6)*exp(0.981327417543519d0*y(5) - 26.7037251572606d0* &
      y(6) - 3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1))
   drdvac(7, 1) = 56583391.0351063d0*y(6)*exp(14.3581999433635d0*y(5))
   drdvac(10, 1) = 37616643.2783057d0

   drdy = drdy + matmul(drdvac, dvacdy)

   jac = matmul(dypdr, drdy) - cj * mas

end subroutine fidadjac

subroutine ratecalc(neqin, yin)

   use solve_ida, only: rates

   implicit none

   integer, intent(in) :: neqin
   real*8, intent(in) :: yin(neqin)
   real*8 :: y(neqin)
   real*8 :: vac(1)

   integer :: i

   y = yin


   vac = 0
   vac(1) = -y(5) - y(6) - y(3) - y(4) - y(2) - y(1) + 1

   do i = 1, 1
      if (vac(i) < -1d-10) then
         vac(i) = 0.d0
      endif
   enddo

   rates = 0
   rates(1) = -1340578.20008235d0*y(3)*y(2)*exp(-15.4081163912808d0*y(5) - 30.7473240605402d0* &
      y(2))*exp(24.8873719748563d0*y(5) + 49.6634417662587d0*y(2)) + &
      63.8798753446516d0*y(1)*vac(1)*exp(-15.4081163912808d0*y(5) - &
      30.7473240605402d0*y(2))
   rates(2) = -66299103748.64d0*y(3)*y(1)*exp(1.97619170776594d0*y(5)) + 261721.165850268d0* &
      y(4)*vac(1)*exp(-6.79547637531067d0*y(5))
   rates(3) = -2.30770198969821d0*y(5)*exp(34.8099614189798d0*y(5) + 42.1998123408802d0* &
      y(6) + 5.2747691352989d0*y(3) + 2.34216336236839d0*y(4) + &
      25.4516651497035d0*y(2) + 5.8390623101461d0*y(1)) + 22338886.1940868d0 &
      *vac(1)
   rates(4) = -1594735764.81432d0*y(5)*y(1)*exp(-1.55078861175432d0*y(5) + &
      42.1998123408802d0*y(6) + 5.2747691352989d0*y(3) + &
      2.34216336236839d0*y(4) + 25.4516651497035d0*y(2) + 5.8390623101461d0 &
      *y(1))*exp(0.981327417543519d0*y(5) - 26.7037251572606d0*y(6) - &
      3.33783439886465d0*y(3) - 1.48210343204541d0*y(4) - &
      16.1056230644397d0*y(2) - 3.69491489314543d0*y(1)) + &
      2686955.14446704d0*y(6)*vac(1)*exp(0.981327417543519d0*y(5) - &
      26.7037251572606d0*y(6) - 3.33783439886465d0*y(3) - &
      1.48210343204541d0*y(4) - 16.1056230644397d0*y(2) - &
      3.69491489314543d0*y(1))
   rates(5) = 10897614040166.0d0*y(6)*y(2)
   rates(6) = 206758.569691054d0*y(5)*y(2)*exp(18.2052444265192d0*y(5) + 12.7487082935497d0* &
      y(6) + 1.59352587823243d0*y(3) + 0.707575598712988d0*y(4) + &
      22.6925318713496d0*y(2) + 1.7640007850888d0*y(1))
   rates(7) = 56583391.0351063d0*y(6)*vac(1)*exp(14.3581999433635d0*y(5))
   rates(8) = 10897614040166.0d0*y(6)*y(1)
   rates(9) = -153192832.848627d0*y(3)**2*exp(10.5495382705978d0*y(5))
   rates(10) = -253630218456.946d0*y(4)*exp(2.34216336236839d0*y(5)) + 37616643.2783057d0* &
      vac(1)
   rates(11) = -1.18757991768423d+26*y(4)*y(2)*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2))*exp(16.1157038917797d0*y(5) + &
      49.6634417662587d0*y(2))/(131544332324398.0d0*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2)) + &
      10897614040166.0d0) + 1.43351936284262d+27*y(1)**2*exp( &
      -16.1157038917797d0*y(5) - 49.6634417662587d0*y(2))/( &
      131544332324398.0d0*exp(-16.1157038917797d0*y(5) - &
      49.6634417662587d0*y(2)) + 10897614040166.0d0)

end subroutine ratecalc

subroutine fidajtimes(tres, yin, ypin, res, vin, fjv, cj, ewt, h, ipar, rpar, wk1, wk2, ier)

   use solve_ida, only: neq

   implicit none

   real*8, intent(in) :: tres, yin(neq), ypin(neq), res(neq), vin(neq), cj, h
   real*8 :: ewt(*), wk1(*), wk2(*), rpar(*), wk3(1)
   integer :: ipar(*)
   integer :: i

   real*8, intent(out) :: fjv(neq)
   integer, intent(out) :: ier

   real*8 :: jac(neq, neq)

   call fidadjac(neq, tres, yin, ypin, res, jac, cj, ewt, h, ipar, rpar, wk1, wk2, wk2, ier)

   fjv = 0.d0

   call dgemv('N', neq, neq, 1.d0, jac, neq, vin, 1, 0.d0, fjv, 1)

end subroutine fidajtimes

subroutine fidapsol(tres, yin, ypin, res, rvin, zv, cj, delta, ewt, ipar, rpar, wk1, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), ypin(neq), res(neq), rvin(neq)
   real*8, intent(in) :: cj, delta, ewt(*), rpar(*)
   integer, intent(in) :: ipar(*)

   real*8 :: wk1(*)
   integer :: ier

   real*8, intent(out) :: zv(neq)

   integer :: ipiv(neq)
   integer :: i

   zv = rvin
   call dgesv(neq, 1, jac, neq, ipiv, zv, neq, ier)

end subroutine fidapsol

subroutine fidapset(tres, yin, ypin, res, cj, ewt, h, ipar, rpar, wk1, wk2, wk3, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), ypin(neq), res(neq)
   real*8, intent(in) :: cj, ewt(*), h, rpar(*)
   real*8 :: wk1(*), wk2(*), wk3(*)

   integer, intent(in) :: ipar(*)

   integer, intent(out) :: ier

   call fidadjac(neq, tres, yin, ypin, res, jac, cj, ewt, h, ipar, rpar, wk1, wk2, wk3, ier)

end subroutine fidapset

   