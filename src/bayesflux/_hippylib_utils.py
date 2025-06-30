import time
import types
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import dolfin as dl
    import hickle
    import hippylib as hp

    _HIPPYLIB_AVAILABLE = True
except ImportError as e:
    _HIPPYLIB_AVAILABLE = False

import numpy as np

from .generation import GaussianInputOuputAndDerivativesSampler

if _HIPPYLIB_AVAILABLE:

    class _ObservableJacobian:
        """
        This class implements matrix free application of the Jacobian operator.
        The constructor takes the following parameters:

        - :code:`model`:               the object which contains the description of the problem.

        Type :code:`help(modelTemplate)` for more information on which methods model should implement.
        """

        def __init__(self, observable):
            """
            Construct the Observable Jacobian operator
            """
            self.observable = observable

            self.ncalls = 0

            self.rhs_fwd = observable.generate_vector(hp.STATE)
            self.rhs_adj = observable.generate_vector(hp.ADJOINT)
            self.rhs_adj2 = observable.generate_vector(hp.ADJOINT)
            self.uhat = observable.generate_vector(hp.STATE)
            self.phat = observable.generate_vector(hp.ADJOINT)
            self.yhelp = observable.generate_vector(hp.PARAMETER)

            self.Bu = dl.Vector()
            self.observable.B.init_vector(self.Bu, 0)
            self.Ctphat = observable.generate_vector(hp.PARAMETER)
            self.shape = (self.Bu.get_local().shape[0], self.yhelp.get_local().shape[0])

            # form C
            n_state = self.rhs_fwd.get_local().shape[0]
            n_obs = self.Bu.get_local().shape[0]
            n_param = self.yhelp.get_local().shape[0]

            # Form C matrix: maps state → param (C_matrix: param x state)
            self.C_vectors = []  # np.zeros((n_state, n_param))
            for i in range(n_param):
                self.yhelp.zero()
                self.yhelp[i] = 1.0
                self.observable.applyC(self.yhelp, self.uhat)  # yhelp = C uhat
                # self.C_matrix[:, i] = self.uhat.get_local()
                self.C_vectors.append(self.uhat.get_local().copy())

            # Form B matrix: maps state → n_obs (B_matrix: obs x state)
            self.B_matrix = np.zeros((n_obs, n_state))
            for i in range(n_state):
                self.uhat.zero()
                self.uhat[i] = 1.0
                # self.Bu.zero()
                self.observable.applyB(self.uhat, self.Bu)  # Bu = B v
                self.B_matrix[:, i] = self.Bu.get_local()

        def mpi_comm(self):
            return self.observable.B.mpi_comm()

        def init_vector(self, x, dim):
            """
            Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
            operator.

            Parameters:

            - :code:`x`: the vector to reshape.
            - :code:`dim`: if 0 then :code:`x` will be made compatible with the range of the Jacobian, if 1 then :code:`x` will be made compatible with the domain of the Jacobian.

            """
            if dim == 0:
                self.observable.init_vector(x, 0)
            elif dim == 1:
                self.observable.init_vector(x, 1)
                # If the prior term shows up then the input dimension changes due to quadrature workaround
                # self.model.prior.sqrtM.init_vector(x,1)
            else:
                raise

        def mult(self, x, y):
            """
            Apply the Jacobian :code:`x`. Return the result in :code:`y`.
            Implemented for dl.Vector
            """
            self.observable.applyC(x, self.rhs_fwd)
            self.observable.solveFwdIncremental(self.uhat, self.rhs_fwd)
            assert hasattr(self.observable, "applyB"), "LinearObservable must have attribute applyB"
            self.observable.applyB(self.uhat, y)
            y *= -1.0
            self.ncalls += 1

        def transpmult(self, x, y):
            """
            Apply the Jacobian transpose :code:`x`. Return the result in :code:`y`.
            Implemented for dl.Vector
            """
            assert hasattr(self.observable, "applyBt"), "LinearObservable must have attribute applyBt"
            self.observable.applyBt(x, self.rhs_adj)
            self.observable.solveAdjIncremental(self.phat, self.rhs_adj)
            self.observable.applyCt(self.phat, y)
            y *= -1.0
            self.ncalls += 1

    class __LinearStateObservable:
        """ """

        def __init__(self, problem, B):
            """
            Create a model given:
                - problem: the description of the forward/adjoint problem and all the sensitivities
                - B: the pointwise observation operator
                - prior: the prior
            """
            self.problem = problem
            # self.is_control_problem = hasattr(self.problem,'Cz')
            self.B = B
            # self.Bu = dl.Vector(self.B.mpi_comm())
            # self.B.init_vector(self.Bu,0)

            self.n_fwd_solve = 0
            self.n_adj_solve = 0
            self.n_inc_solve = 0

            if hasattr(self.problem, "parameter_projection"):
                self.out1 = self.problem.generate_parameter()

        def mpi_comm(self):
            return self.B.mpi_comm()

        def generate_vector(self, component="ALL"):
            """
            By default, return the list :code:`[u,m,p]` where:
            
                - :code:`u` is any object that describes the state variable
                - :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
                (Needs to support linear algebra operations)
                - :code:`p` is any object that describes the adjoint variable
            
            If :code:`component = STATE` return only :code:`u`
                
            If :code:`component = PARAMETER` return only :code:`m`
                
            If :code:`component = ADJOINT` return only :code:`p`
            """
            if component == "ALL":
                x = [self.problem.generate_state(), self.problem.generate_parameter(), self.problem.generate_state()]
            elif component == hp.STATE:
                x = self.problem.generate_state()
            elif component == hp.PARAMETER:
                x = self.problem.generate_parameter()
            elif component == hp.ADJOINT:
                x = self.problem.generate_state()

            return x

        def init_vector(self, x, dim):
            """
            Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
            operator.

            Parameters:

            - :code:`x`: the vector to reshape.
            - :code:`dim`: if 0 then :code:`x` will be made compatible with the range of the Jacobian, if 1 then :code:`x` will be made compatible with the domain of the Jacobian.

            """
            if dim == 0:
                self.B.init_vector(x, 0)
            elif dim == 1:
                # self.model.init_parameter(x)
                self.problem.C.init_vector(x, 1)
            else:
                raise

        def init_parameter(self, m):
            """
            Reshape :code:`m` so that it is compatible with the parameter variable
            """
            # Aha! I found the issue I think!!!!!!!!!
            # This is wrong since the STATE and PARAMETER dimension are not necessarily the same.

            self.problem.init_parameter(m)

        def eval(self, m, u0=None, z=None, setLinearizationPoint=False):
            """
            Given the input parameter :code:`m` solve for the state field $u(m)$, and evaluate
            the linear state observable $Bu(m)$

            Return the linear state observable $Bu(m)

            """
            if u0 is None:
                u0 = self.problem.generate_state()
            # if self.is_control_problem:
            #     assert z is not None
            #     x = [u0,m,None,z]
            # else:
            x = [u0, m, None]
            self.problem.solveFwd(u0, x)
            if setLinearizationPoint:
                self.setLinearizationPoint(x)
            out = dl.Vector()  # self.mpi_comm()
            self.B.init_vector(out, 0)
            self.B.mult(u0, out)

            return out

        def evalu(self, u, output_vec):
            """
            Given a state field that is already solved for :code:`u`, evaluate the linear
            state observable $Bu(m)$

            Return the linear state observable $Bu(m)

            """
            output_vec.zero()
            # out = dl.Vector(self.mpi_comm())
            # self.B.init_vector(output_vec,0)
            self.B.mult(u, output_vec)
            return output_vec

        def solveFwd(self, out, x):
            """
            Solve the (possibly non-linear) forward problem.

            Parameters:
                - :code:`out`: is the solution of the forward problem (i.e. the state) (Output parameters)
                - :code:`x = [u,m,p]` provides
                    1) the parameter variable :code:`m` for the solution of the forward problem
                    2) the initial guess :code:`u` if the forward problem is non-linear

                    .. note:: :code:`p` is not accessed.
            """
            self.n_fwd_solve = self.n_fwd_solve + 1
            self.problem.solveFwd(out, x)

        def setLinearizationPoint(self, x):
            """
            Specify the point :code:`x = [u,m,p]` at which the Hessian operator (or the Gauss-Newton approximation)
            needs to be evaluated.
            Parameters:
                - :code:`x = [u,m,p]`: the point at which the Hessian or its Gauss-Newton approximation needs to be evaluated.

            .. note:: This routine should either:
                - simply store a copy of x and evaluate action of blocks of the Hessian on the fly
                - or partially precompute the block of the hessian (if feasible)
            """
            x[hp.ADJOINT] = self.problem.generate_state()
            self.problem.setLinearizationPoint(x, True)

        def solveFwdIncremental(self, sol, rhs):
            """
            Solve the linearized (incremental) forward problem for a given right-hand side
            Parameters:
                - :code:`sol` the solution of the linearized forward problem (Output)
                - :code:`rhs` the right hand side of the linear system
            """
            self.n_inc_solve = self.n_inc_solve + 1
            self.problem.solveIncremental(sol, rhs, False)

        def solveAdjIncremental(self, sol, rhs):
            """
            Solve the incremental adjoint problem for a given right-hand side
            Parameters:
                - :code:`sol` the solution of the incremental adjoint problem (Output)
                - :code:`rhs` the right hand side of the linear system
            """
            self.n_inc_solve = self.n_inc_solve + 1
            self.problem.solveIncremental(sol, rhs, True)

        def applyB(self, x, out):
            self.B.mult(x, out)

        def applyBt(self, x, out):

            self.B.transpmult(x, out)

        def applyC(self, dm, out):
            """

            Apply the :math:`C` block of the Hessian to a (incremental) parameter variable, i.e.
            :code:`out` = :math:`C dm`

            Parameters:
                - :code:`dm` the (incremental) parameter variable
                - :code:`out` the action of the :math:`C` block on :code:`dm`

            .. note:: This routine assumes that :code:`out` has the correct shape.
            """
            if hasattr(self.problem, "parameter_projection"):
                dm1 = self.problem.parameter_projection(dm)
                self.problem.apply_ij(hp.ADJOINT, hp.PARAMETER, dm1, out)
            else:
                self.problem.apply_ij(hp.ADJOINT, hp.PARAMETER, dm, out)

        def applyCt(self, dp, out):
            """
            Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
            :code:`out` = :math:`C^t dp`
            Parameters:
                - :code:`dp` the (incremental) adjoint variable
                - :code:`out` the action of the :math:`C^T` block on :code:`dp`

            ..note:: This routine assumes that :code:`out` has the correct shape.
            """
            if hasattr(self.problem, "parameter_projection"):
                self.out1.zero()
                self.problem.apply_ij(hp.PARAMETER, hp.ADJOINT, dp, self.out1)
                self.problem.transmult_M(self.out1, out)
            else:
                self.problem.apply_ij(hp.PARAMETER, hp.ADJOINT, dp, out)

    def __hippylibModelLinearStateObservable(model):
        """
        This function construcst a linear state observable from
        hIPPYlib.modeling.model.Model attributes
        Parameters:
            - :code:`model` represents the hippylib mode
        """
        assert hasattr(model, "misfit")
        return __LinearStateObservable(model.problem, model.misfit.B)

    def mv_to_dense_local(multivector, out_array=None):
        """
        This function converts a MultiVector object to a numpy array
            - :code:`multivector` - hippylib MultiVector object
        """
        multivector_shape = (multivector[0].get_local().shape[0], multivector.nvec())
        if out_array is None:
            out_array = np.zeros(multivector_shape)
        for i in range(multivector_shape[-1]):
            out_array[:, i] = multivector[i].get_local()

        return out_array

    def mv_to_dense(multivector, out_array=None):
        """
        This function converts a MultiVector object to a numpy array
            - :code:`multivector` - hippylib MultiVector object
        """
        multivector_shape = (multivector[0].gather_on_zero().shape[0], multivector.nvec())
        if out_array is None:
            out_array = np.zeros(multivector_shape)
        for i in range(multivector_shape[-1]):
            out_array[:, i] = multivector[i].gather_on_zero()

        return out_array

    def dense_to_mv_local(dense_array, dl_vector):
        """
        This function converts a numpy array to a MultiVector
            - :code:`dense_array` - numpy array to be transformed
            - :code:`dl_vector` - type :code:`dolfin.Vector` object to be used in the
                MultiVector object constructor
        """
        # This function actually makes no sense
        temp = hp.MultiVector(dl_vector, dense_array.shape[-1])
        for i in range(temp.nvec()):
            temp[i].set_local(dense_array[:, i])
        return temp

    def _dense_prior_precision_and_mass_matrix_from_hippylib_prior(hippylib_BiHarmonicCovariancePrior):
        # Extract BiLaplacianPrior parameters as discrete matrices, so that we don't rely on hippylib afterwards
        import scipy.sparse as ss

        M_mat = dl.as_backend_type(hippylib_BiHarmonicCovariancePrior.M).mat()
        row, col, val = M_mat.getValuesCSR()
        M = ss.csr_matrix((val, col, row)).tocsc().todense()

        A_mat = dl.as_backend_type(hippylib_BiHarmonicCovariancePrior.A).mat()
        row, col, val = A_mat.getValuesCSR()
        A = ss.csr_matrix((val, col, row)).todense()
        return A, M

    class __hippylibObservableWrapper(GaussianInputOuputAndDerivativesSampler):
        def __init__(self, model, observable, random_seed):  # relies on observable Jacobian!
            # iniitalize variables
            self.problem = model.problem
            self._u = observable.generate_vector(hp.STATE)
            self._m = observable.generate_vector(hp.PARAMETER)
            self._output = dl.Vector()
            observable.B.init_vector(self._output, 0)
            self._noise = dl.Vector()
            self._model = model
            self._model.prior.init_vector(self._noise, "noise")
            self._noise_precision = self._model.misfit.noise_precision
            self._noise_variance = self._model.misfit.noise_variance
            self.observable = observable
            self._J = _ObservableJacobian(observable)
            self.non_parallel_random_generator = hp.Random(seed=random_seed)

            # required attributes
            self._A, self._M = _dense_prior_precision_and_mass_matrix_from_hippylib_prior(model.prior)
            self.__L2_inner_product_matrix, self.__precision = None, None

            self._input_dimension = self._m.get_local().shape[0]
            self._output_dimension = self._output.get_local().shape[0]

        @property
        def _precision(self):
            if self.__precision == None:
                import jax

                jax.config.update("jax_enable_x64", True)
                import jax.numpy as jnp

                self._A = jax.device_put(self._A)
                self.__L2_inner_product_matrix = jax.device_put(self._M)
                self.__precision = self._A @ (jnp.linalg.solve(self.__L2_inner_product_matrix, self._A))
                print("computed precision")
                return self.__precision
            else:
                return self.__precision

        @property
        def _L2_inner_product_matrix(self):
            if self.__L2_inner_product_matrix == None:
                import jax

                jax.config.update("jax_enable_x64", True)
                self.__L2_inner_product_matrix = jax.device_put(self._M)
                print("got here")
                return self.__L2_inner_product_matrix
            else:
                return self.__L2_inner_product_matrix

        def sample_input(self) -> np.ndarray:
            self._noise.zero()
            self.non_parallel_random_generator.normal(1, self._noise)
            self._m.zero()
            self._model.prior.sample(self._noise, self._m)
            # plt.figure()
            # p = dl.plot(vector2Function(self._m, self._model.problem.Vh[PARAMETER]))
            # p.set_cmap("viridis")
            # plt.imshow( fenics_vector_to_grid(self._model.problem.Vh[PARAMETER],self._m.get_local()),interpolation='bilinear', cmap='Blues') #[(250,100)]
            # plt.axis('off')
            # plt.show()
            return self._m.get_local()

        def _init_value(self): ...

        def _init_matrix_jacobian_prod(self, *, matrix: np.ndarray = None):
            dQr = matrix.shape[1] if matrix is not None else self.output_dimension
            self._output_reduced_jacobian_transpose = np.zeros((self._input_dimension, dQr))
            self._OutputMV = (
                dense_to_mv_local(matrix, self._output)
                if matrix is not None
                else dense_to_mv_local(np.identity(self.output_dimension), self._output)
            )
            self._JstarMatMV = hp.MultiVector(self._m, dQr)
            # We use these data structures as temporary storage when evaluating the matrix jacobian products

        def _value(self, input_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            self._m.set_local(input_sample)
            x = [self._u, self._m, None]
            self.observable.solveFwd(self._u, x)
            output_sample = self._J.observable.evalu(self._u, self._output).get_local()
            return output_sample

        def value_and_matrix_jacobian_prod(self, input_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            self._m.set_local(input_sample)
            x = [self._u, self._m, None]
            start = time.perf_counter()
            self._J.observable.solveFwd(self._u, x)
            output_sample = self._J.observable.evalu(self._u, self._output).get_local()
            self.output_computation_time += time.perf_counter() - start
            # print("Fwd time:", time_for_sample, flush=True)
            start = time.perf_counter()
            self._J.observable.setLinearizationPoint(x)
            self._JstarMatMV.zero()
            hp.MatMvTranspmult(self._J, self._OutputMV, self._JstarMatMV)  # J^T OutputEncoder
            mv_to_dense(self._JstarMatMV, self._output_reduced_jacobian_transpose)  # output basis
            time_for_sample = time.perf_counter() - start
            self.jacobian_product_computation_time += time_for_sample
            # print("vjp time:", time_for_sample, flush=True)
            return output_sample, self._output_reduced_jacobian_transpose.T

    def hippylib_sampler_from_model(model_name, seed):
        import importlib

        model_module = importlib.import_module(model_name)

        # DEFINE  settings
        settings = model_module.settings()
        settings["hippylib_seed"] = seed

        model = model_module.model(settings)

        sampler_wrapper = __hippylibObservableWrapper(
            model, __hippylibModelLinearStateObservable(model), settings["hippylib_seed"]
        )
        return sampler_wrapper

    def __worker_generates_reduced_hippylib_training_data(
        task_id: int, *, model_name: str, N_samples: int, random_seed_offset: int, **kwargs
    ):
        from bayesflux.generation import generate_reduced_training_data

        # if task_id == 0: print("loading hippylib model on process 0")
        worker_sampler_wrapper = hippylib_sampler_from_model(model_name, task_id + random_seed_offset)
        # if task_id == 0: print("done loading hippylib model on process 0")

        reduced_dims = kwargs.get("reduced_dims")
        if reduced_dims:
            kwargs.pop("reduced_dims")
            encodec_dict = hickle.load(kwargs.pop("encodec_path"))[reduced_dims]
            kwargs["input_decoder"], kwargs["input_encoder"], kwargs["output_encoder"] = (
                encodec_dict["input"]["decoder"],
                encodec_dict["input"]["encoder"],
                encodec_dict["output"]["encoder"],
            )

        results = generate_reduced_training_data(sampler_wrapper=worker_sampler_wrapper, N_samples=N_samples, **kwargs)
        from pathlib import Path

        Path("./multiprocess_tmp").mkdir(parents=True, exist_ok=True)
        hickle.dump(results, f"./multiprocess_tmp/{task_id}.hkl", mode="w")
        return task_id

    def __single_process_worker_generates_reduced_hippylib_training_data(
        task_id: int, *, model_name: str, N_samples: int, N_processes: int, random_seed_offset: int, **kwargs
    ):
        # 1) force one thread
        import os

        import psutil

        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = "1"
        # 2) pin CPU
        psutil.Process(os.getpid()).cpu_affinity([task_id])

        # 3) assign number of samples for each task_id/worker
        quotient, remainder = divmod(N_samples, N_processes)
        N_samples_on_processes = [quotient + (1 if i < remainder else 0) for i in range(N_processes)]

        # load encoder/decoders
        return __worker_generates_reduced_hippylib_training_data(
            task_id,
            model_name=model_name,
            N_samples=N_samples_on_processes[task_id],
            random_seed_offset=random_seed_offset,
            **kwargs,
        )

    def __single_process_worker_generator(model_name, N_samples, N_processes, random_seed_offset, **kwargs):
        from functools import partial

        return partial(
            __single_process_worker_generates_reduced_hippylib_training_data,
            model_name=model_name,
            N_samples=N_samples,
            N_processes=N_processes,
            random_seed_offset=random_seed_offset,
            **kwargs,
        )

    def __multiprocess(worker_fn, N_processes):
        ctx = get_context("spawn")
        with ctx.Pool(processes=N_processes) as pool:
            results = pool.map(worker_fn, list(range(N_processes)))
        return results

    def multiprocess_generate_reduced_hippylib_training_data(
        model_name: str, N_samples: int, N_processes: int, random_seed_offset: int = 0, **kwargs
    ):
        # generate training data in parallel using different hippylib seeds on each processes, save to disk
        __multiprocess(
            __single_process_worker_generator(
                model_name, N_samples, N_processes, random_seed_offset=random_seed_offset, **kwargs
            ),
            N_processes,
        )

        # load results of each process from file
        # print("Loading results from files")
        import time

        start = time.perf_counter()
        N_results_dcts = [hickle.load(f"./multiprocess_tmp/{i}.hkl") for i in range(N_processes)]

        # Concatenate results, add metadata, and delete temporary stored files, and return results
        # print("Accumulating generated training data to parent process")
        accumulated_results = dict()
        for k in N_results_dcts[0].keys():
            if isinstance(N_results_dcts[0][k], np.ndarray) and N_results_dcts[0][k].shape != ():
                accumulated_results[k] = np.concatenate([d[k] for d in N_results_dcts], axis=0)
            else:
                times = np.array([d[k] for d in N_results_dcts])
                accumulated_results[k] = np.sum(times)
                accumulated_results["parallel_max_" + k] = np.max(times)
                accumulated_results[k.removesuffix("_time") + "_num_parallel_processes"] = N_processes
                # print("parallel_max_" + k, accumulated_results["parallel_max_" + k])
        accumulation_to_parent_process_time = time.perf_counter() - start
        accumulated_results["accumulation_to_parent_process_time"] = accumulation_to_parent_process_time
        # print("accumulation_to_parent_process_time", accumulation_to_parent_process_time)
        import shutil

        shutil.rmtree("multiprocess_tmp")
        return accumulated_results

    def multiprocess_generate_hippylib_output_data(
        *,
        model_name: str,
        N_samples: int,
        N_processes: int,
        random_seed_offset: int = 0,
        print_progress=False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate output data for dimension reduction.

        This is a thin wrapper around generate_reduced_training_data, which only
        requires an object that accepts NumPy arrays as input and returns NumPy arrays.

        Parameters:
            model_name: Name of hte model which will be instantiated via hippylib_utils
            N_samples: The number of samples to generate.

        Returns:
            A dictionary containing training data and optional computation times.
        """
        return multiprocess_generate_reduced_hippylib_training_data(
            model_name=model_name,
            N_samples=N_samples,
            N_processes=N_processes,
            random_seed_offset=random_seed_offset,
            generate_Jacobians=False,
            print_progress=print_progress,
            **kwargs,
        )

    def multiprocess_generate_hippylib_full_Jacobian_data(
        *, model_name: str, N_samples: int, N_processes: int, random_seed_offset: int = 0, print_progress: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate full Jacobian data for dimension reduction.

        This is a thin wrapper around generate_reduced_training_data, which only
        requires an object that accepts NumPy arrays as input and returns NumPy arrays.

        Parameters:
            model_name: Name of hte model which will be instantiated via hippylib_utils
            N_samples: The number of samples to generate.

        Returns:
            A dictionary containing training data and optional computation times.
        """
        return multiprocess_generate_reduced_hippylib_training_data(
            model_name=model_name,
            N_samples=N_samples,
            N_processes=N_processes,
            random_seed_offset=random_seed_offset,
            generate_Jacobians=True,
            print_progress=print_progress,
        )

    ##########################
    # log likelihood methods #
    ##########################
    def _create_vector(self, dim):
        new_vector = dl.Vector()
        self.init_vector(new_vector, dim)
        return new_vector

    def _create_vector_method(hp_ModelInstance: hp.Model):
        hp_ModelInstance.create_vector = types.MethodType(_create_vector, hp_ModelInstance)
        return hp_ModelInstance

    def NewVectorFromFunctionSpace(Vh: dl.FunctionSpace):
        return dl.Function(Vh).vector()

    def np_to_dl(np_array: np.ndarray, dl_vector) -> List[dl.Vector]:
        dl_vector.set_local(np_array)

    def _log_likelihood_joint_numpy(
        self,
        encoded_Y: np.ndarray,
        encoded_X: np.ndarray,
        input_decoder: np.ndarray,
        output_encoder: np.ndarray,
        output_decoder_T: np.ndarray,
    ):
        N = encoded_Y.shape[0]
        log_likes = np.zeros(N)
        input_decoder_T = input_decoder.T.copy()

        for i, (X_i, Y_i) in enumerate(zip(encoded_X, encoded_Y)):
            print("log like i", i)
            np_to_dl(X_i @ input_decoder_T, self.dl_X_vector)
            ump = [self.u, self.dl_X_vector, self.p]
            try:
                self.model.misfit.d.set_local(Y_i @ output_decoder_T)
                self.model.solveFwd(ump[hp.STATE], ump)
            except:
                raise ("Forward Solve failed")
            log_likes[i] = -self.model.misfit.cost(ump)
        return log_likes

    def _value_and_grad_log_likelihood_joint_numpy(
        self,
        encoded_Y: np.ndarray,
        encoded_X: np.ndarray,
        input_decoder: np.ndarray,
        output_encoder: np.ndarray,
        output_decoder_T: np.ndarray,
    ):
        # allocate array of scalars, array of vectors
        N = encoded_Y.shape[0]
        log_likes = np.zeros(N)
        gradients = np.zeros((N, encoded_X.shape[1]))
        self.model.misfit.projector = output_decoder_T.T @ output_encoder.T
        self.model.misfit.output_encoder = output_encoder

        for i, (X_i, Y_i) in enumerate(zip(encoded_X, encoded_Y)):
            np_to_dl(input_decoder @ X_i, self.dl_X_vector)  # reuse one dolfin vector
            ump = [self.u, self.dl_X_vector, self.p]
            try:
                self.model.solveFwd(ump[hp.STATE], ump)  #  \| D_y \cdot \|^2_{Gamma^-1} = \| \cdot \|^2?
                self.model.misfit.d.set_local(Y_i @ output_decoder_T)
                self.model.solveAdj(ump[hp.ADJOINT], ump)
                grad_norm = self.model.evalGradientParameter(ump, self.mg_vector, misfit_only=True)
                log_likes[i] = -self.model.misfit.cost(ump)
                gradients[i] = -(self.mg_vector.get_local()) @ input_decoder
            except:
                print("Gradient evaluation failed, returning 0 vector for the gradient")
                self.mg_vector.zero()
                log_likes[i] = -1.0e4
                gradients[i] = -self.mg_vector.get_local() @ input_decoder
        return log_likes, gradients

    def __add_numpy_joint_methods(hp_MisfitInstance: hp.Misfit):
        hp_MisfitInstance.log_prob = types.MethodType(_log_likelihood_joint_numpy, hp_MisfitInstance)
        hp_MisfitInstance.value_and_grad_log_prob = types.MethodType(
            _value_and_grad_log_likelihood_joint_numpy, hp_MisfitInstance
        )

        return hp_MisfitInstance

    def __prepare_hippylib_misfit(hippylib_posterior):
        hippylib_posterior.misfit.u = hippylib_posterior.problem.generate_state()
        hippylib_posterior.misfit.p = NewVectorFromFunctionSpace(hippylib_posterior.problem.Vh[hp.STATE])
        hippylib_posterior.prior = _create_vector_method(hippylib_posterior.prior)
        hippylib_posterior.misfit.prior = hippylib_posterior.prior
        hippylib_posterior.misfit.dl_X_vector = hippylib_posterior.misfit.prior.create_vector(0)
        hippylib_posterior.misfit.mg_vector = hippylib_posterior.generate_vector(hp.PARAMETER)
        misfit_copy = hippylib_posterior.misfit
        misfit_copy.model = hippylib_posterior
        return misfit_copy

    def hippylib_loglikelihood_from_model(model_name):
        import importlib

        model_module = importlib.import_module(model_name)

        model = model_module.model(model_module.settings())  # does this have the right misfit???
        return __add_numpy_joint_methods(__prepare_hippylib_misfit(model))

    def __worker_evaluates_log_likelihood_and_gradient(
        task_id: int, *, model_name: str, encodec_path: Path, reduced_dims: Tuple[int, int], **kwargs
    ):
        log_likelihood_wrapper = hippylib_loglikelihood_from_model(model_name)
        # time this!
        encodec_dict = hickle.load(encodec_path)[reduced_dims]  # cpu
        input_decoder, output_encoder, output_decoder_T = (
            encodec_dict["input"]["decoder"],
            encodec_dict["output"]["encoder"],
            encodec_dict["output"]["decoder"].T,
        )
        encoded_Ys, encoded_Xs = np.load(f"./multiprocess_encoded_Y_tmp/{task_id}.npy"), np.load(
            f"./multiprocess_encoded_X_tmp/{task_id}.npy"
        )
        import time

        results = dict()
        start = time.perf_counter()
        results["values"], results["gradients"] = log_likelihood_wrapper.value_and_grad_log_prob(
            encoded_Ys, encoded_Xs, input_decoder, output_encoder, output_decoder_T
        )
        results["values_and_gradient_computation_time"] = time.perf_counter() - start

        from pathlib import Path

        Path("./multiprocess_loglike_tmp").mkdir(parents=True, exist_ok=True)
        hickle.dump(results, f"./multiprocess_loglike_tmp/{task_id}.hkl", mode="w")

        return task_id

    def __single_process_worker_evaluates_log_likelihood_and_gradient(
        task_id: int, *, model_name: str, encodec_path: Path, reduced_dims: Tuple[int, int], **kwargs
    ):
        # 1) force one thread
        import os

        import psutil

        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = "1"
        # 2) pin CPU
        psutil.Process(os.getpid()).cpu_affinity([task_id])

        return __worker_evaluates_log_likelihood_and_gradient(
            task_id,
            model_name=model_name,
            encodec_path=encodec_path,
            reduced_dims=reduced_dims,
            **kwargs,
        )

    def __single_process_worker_evaluate_log_likelihood_and_gradient(
        model_name, encodec_path: Path, reduced_dims: Tuple[int, int], **kwargs
    ):
        from functools import partial

        return partial(
            __single_process_worker_evaluates_log_likelihood_and_gradient,
            model_name=model_name,
            encodec_path=encodec_path,
            reduced_dims=reduced_dims,
            **kwargs,
        )

    def multiprocess_evaluate_encoded_hippylib_log_likelihood_and_gradient(
        model_name: str,
        encoded_X: np.ndarray,
        encoded_Y: np.ndarray,
        encodec_path: Path,
        reduced_dims: Tuple[int, int],
        N_processes: int,
        **kwargs,
    ):
        # split X/Y into chunks to process on different processes and save to individual hickle files
        quotient, remainder = divmod(encoded_X.shape[0], N_processes)
        N_samples_on_processes = [quotient + (1 if i < remainder else 0) for i in range(N_processes)]
        from pathlib import Path

        Path("./multiprocess_encoded_X_tmp").mkdir(parents=True, exist_ok=True)
        Path("./multiprocess_encoded_Y_tmp").mkdir(parents=True, exist_ok=True)
        start_idx = 0
        for i, N_samples_on_process_i in enumerate(N_samples_on_processes):
            np.save(f"./multiprocess_encoded_X_tmp/{i}.npy", encoded_X[start_idx : start_idx + N_samples_on_process_i])
            np.save(f"./multiprocess_encoded_Y_tmp/{i}.npy", encoded_Y[start_idx : start_idx + N_samples_on_process_i])
            start_idx = start_idx + N_samples_on_process_i
        # generate training data in parallel using different hippylib seeds on each processes, save to disk
        __multiprocess(
            __single_process_worker_evaluate_log_likelihood_and_gradient(
                model_name, encodec_path, reduced_dims, **kwargs
            ),
            N_processes,
        )

        import time

        start = time.perf_counter()
        N_results_dcts = [hickle.load(f"./multiprocess_loglike_tmp/{i}.hkl") for i in range(N_processes)]

        # Concatenate results, add metadata, and delete temporary stored files, and return results
        # print("Accumulating log likelihood gradients to parent process")
        accumulated_results = dict()
        for k in N_results_dcts[0].keys():
            if isinstance(N_results_dcts[0][k], np.ndarray) and N_results_dcts[0][k].shape != ():
                accumulated_results[k] = np.concatenate([d[k] for d in N_results_dcts], axis=0)
            else:
                times = np.array([d[k] for d in N_results_dcts])
                accumulated_results[k] = np.sum(times)
                accumulated_results["parallel_max_" + k] = np.max(times)
                accumulated_results[k.removesuffix("_time") + "_num_parallel_processes"] = N_processes
                # print("parallel_max_" + k, accumulated_results["parallel_max_" + k])
        accumulation_to_parent_process_time = time.perf_counter() - start
        accumulated_results["accumulation_to_parent_process_time"] = accumulation_to_parent_process_time
        import shutil

        shutil.rmtree("multiprocess_encoded_X_tmp")
        shutil.rmtree("multiprocess_encoded_Y_tmp")
        shutil.rmtree("multiprocess_loglike_tmp")

        return accumulated_results

else:

    def _missing_dependency_error(*args, **kwargs):
        raise ImportError(
            "hippylib and FEniCS dependencies are missing. Please install them with:\n"
            "conda create -n fenics-2019.1 -c conda-forge fenics==2019.1.0\n"
            "conda activate fenics-2019.1\n"
            "pip install hippylib bayesflux[hippylib]"
        )

    hippylib_sampler_from_model = _missing_dependency_error
    multiprocess_generate_hippylib_full_Jacobian_data = _missing_dependency_error
    multiprocess_generate_hippylib_output_data = _missing_dependency_error
    multiprocess_generate_reduced_hippylib_training_data = _missing_dependency_error
    multiprocess_evaluate_encoded_hippylib_log_likelihood_and_gradient = _missing_dependency_error
