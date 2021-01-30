from abc import ABC, abstractmethod
import numpy as np
import sympy as sym

class IncompatibleElementTypeError(Exception): pass
class NotInError(Exception): pass
class NdimMismatchedError(Exception): pass
class ElementTypeMismatchedError(Exception): pass
class CauchySequenceNoConvergenceError(Exception): pass
class DiscontinuousFunctionError(Exception): pass
class ValidationError(Exception): pass
class OutOfIntervalError(Exception): pass
class NonSquareIntegrableFunctionError(Exception): pass
class LinearFormConditionsUnsatisfiedError(Exception): pass

# 基礎要素クラス
class AnyElement(ABC):
    def __init__(self, value, symbol=None):
        super().__init__()
        self.symbol = symbol
        self.value = value
        self.validate()
        
    @abstractmethod
    def validate(self):
        pass
    
    def __eq__(self, other):
        return self.value.__eq__(other.value)    
    
    def __repr__(self):
        return f"{type(self).__name__}({self.value})"
    
    def _operate(self, other, operator):
        operate_func = getattr(self.value, operator, None)
        return self.__class__(operate_func(self._get_primitive_value(other)))
    
    def _get_primitive_value(self, x):
        if isinstance(x, AnyElement):
            return x.value
        return x
    
# 基礎集合クラス    
class MathSet():
    def __init__(self, element_builder, fixed_elements=None):
        self.build_element = element_builder
        self.fixed_elements = fixed_elements
    
    def take(self, *args, symbols=[]):
        if self.fixed_elements is not None:
            fiexed_values = [el.value for el in self.fixed_elements]
            for x in args:
                if x in fiexed_values:
                    continue
                raise NotInError(f"{x} is not in the Set")
                
            outs = [self.fixed_elements[idx] for idx in [fiexed_values.index(x) for x in args]]
        else:
            if len(symbols) > 0:
                outs = list(map(lambda x: self.build_element(x[0], x[1]), zip(args, symbols)))
            else:
                outs = list(map(lambda x: self.build_element(x), args))
        
        if len(outs) == 1:
            return outs[0]
        
        return tuple(outs)
    
    def subset(self, X):
        if type(X) != list:
            raise IncompatibleElementTypeError(f"X should be a list or numpy.ndarray. not <{type(x).__name__}>")
        
        return MathSet(self.build_element, list(map(lambda x: self.build_element(x), X)))
    
# 実数の要素クラス
class R(AnyElement):
    def validate(self):
        # 少数、整数以外はIncompatibleElementTypeErrorをraiseする
        if type(self.value) is not float and type(self.value) is not int:
            message = f"element type should be a float or int. not <{type(self.value).__name__}>"
            raise IncompatibleElementTypeError(message)
    
    def __radd__(self, other):
        return self._operate(other, "__radd__")
            
    def __add__(self, other):
        return self._operate(other, "__add__")
    
    def __rsub__(self, other):
        return self._operate(other, "__rsub__")
           
    def __sub__(self, other):
        return self._operate(other, "__sub__")
    
    def __rmul__(self, other):
        return self._operate(other, "__rmul__")
    
    def __mul__(self, other):
        return self._operate(other, "__mul__")
    
    def __truediv__(self, other):
        return self._operate(other, "__truediv__")
        
    def __rtruediv__(self, other):
        return self._operate(other, "__rtruediv__")
    
    def __lt__(self, other):
        return self.value.__lt__(other.value)

# 抽象ベクトルクラス
class Vector(AnyElement):
    def __radd__(self, other):
        return self._operate(other, "__radd__")
            
    def __add__(self, other):
        return self._operate(other, "__add__")
    
    def __rsub__(self, other):
        return self._operate(other, "__rsub__")
           
    def __sub__(self, other):
        return self._operate(other, "__sub__")
    
    def __rmul__(self, other):
        return self._operate(other, "__rmul__")
    
    def __mul__(self, other):
        return self._operate(other, "__mul__")
    
    def __truediv__(self, other):
        return self._operate(other, "__truediv__")
        
    def __rtruediv__(self, other):
        return self._operate(other, "__rtruediv__")
    
# 実ベクトルクラス
class RealVector(Vector):
    def __init__(self, value, ndim):
        self.ndim = ndim
        super().__init__(value)
    
    def validate(self):
        if type(self.value) is not np.ndarray:
            raise IncompatibleElementTypeError(f"element type should be a numpy.ndarray. not <{type(self.value).__name__}>")        
        
        # ベクトルの成分が実数であるか
        if self.value.dtype not in [np.float32, np.float64, np.int32, np.int64]:
            raise IncompatibleElementTypeError(f"the input vector contains non real number element")
            
        if self.ndim != self.value.shape[0]:
            raise NdimMismatchedError(f"n-dim mismatched: {self.ndim}, {self.value.shape[0]}")
    
    def _operate(self, other, operator):
        operate_func = getattr(self.value, operator, None)
        return self.__class__(operate_func(self._get_primitive_value(other)), ndim=self.ndim)
    
# 軽量ベクトル空間クラス    
class MetricVectorSpace(MathSet):
    def __init__(self, VectorType, ndim):
        self.ndim = ndim
        self.VectorType = VectorType
        super().__init__(lambda x: VectorType(x, ndim=ndim))
    
    def dot(self, x, y):
        if self.ndim != x.ndim:
            raise NdimMismatchedError(f"n-dim mismatched. x dim should be {self.ndim}")
            
        if self.ndim != y.ndim:
            raise NdimMismatchedError(f"n-dim mismatched. y dim should be {self.ndim}")
            
        if type(x) != self.VectorType:
            raise ElementTypeMismatchedError(f"type of x should be {self.VectorType.__name__}")
            
        if type(y) != self.VectorType:
            raise ElementTypeMismatchedError(f"type of y should be {self.VectorType.__name__}")
            
        return sum(map(lambda pair: np.conjugate(pair[0]) * pair[1], zip(x.value, y.value)))
    
    def norm(self, x):
        return np.real(np.sqrt(self.dot(x, x)))
    
    def distance(self, x, y):
        return self.norm(x - y)
    
class AutomatedTestable(ABC):
    @abstractmethod
    def zeros(self): pass
    
    @abstractmethod
    def random_scalar(self): pass
    
    @abstractmethod
    def random_take(self): pass
    
def test_norm_conditions_fulfilled(S):
    # (N 1)
    x = S.random_take()
    assert(S.norm(x) >= 0)
    
    zero = S.zeros()
    assert(S.norm(zero) == 0)
    
    # (N 2)
    a = S.random_scalar()
    x = S.random_take()
    assert(S.norm(a * x) == abs(a) * S.norm(x))
    
    # (N 3)
    x = S.random_take()
    y = S.random_take()
    assert(S.norm(x + y) <= S.norm(x) + S.norm(y))
    
def test_metric_conditions_fulfilled(S):
    # (M 1)
    x = S.random_take()
    y = S.random_take()
    assert(S.distance(x, y) >= 0)
    assert(S.distance(x, x) == 0)
    assert(S.distance(y, y) == 0)
    
    # (M 2)
    x = S.random_take()
    y = S.random_take()
    assert(S.distance(x, y) == S.distance(y, x))
    
    # (M 3)
    x = S.random_take()
    y = S.random_take()
    z = S.random_take()
    assert(S.distance(x, y) <= S.distance(x, z) + S.distance(z, y))
    
# ノルム空間クラス    
class NormSpace(MathSet, ABC):
    @abstractmethod
    def norm(self, f, symbol=None):
        pass
    
# 距離空間クラス
class MetricSpace(MathSet, ABC):
    @abstractmethod
    def distance(self, f, g, symbol=None):
        pass
        
# バナッハ空間クラス        
class BanachSpace(NormSpace, MetricSpace, ABC):
    def check_pseudo_cauchy_sequence_convergence(self, sequence, x_symbol, n_symbol, interval, N, epsilon):
        M = 10000
        N = np.random.randint(N, N+M)
        n_range = np.arange(N, N+M, 10)
        m_range = np.arange(N+1, N+M, 10)
        
        # コーシー列の収束判定
        for m, n in zip(m_range, n_range):
            for x in np.arange(float(interval.inf), float(interval.sup+1e-1), 1e-1):
                f_m = sequence.evalf(subs={x_symbol: x, n_symbol: m})
                f_n = sequence.evalf(subs={x_symbol: x, n_symbol: n})
                distance = self.norm(f_m - f_n)
                if distance > epsilon:
                    raise CauchySequenceNoConvergenceError(f"{distance} > {epsilon}, x={x}, m={m}, n={n}")
                    
        # 関数列の極限が収束するか
        if sym.limit(sequence, n_symbol, sym.oo).has(sym.oo, -sym.oo, sym.zoo, sym.nan):
            raise CauchySequenceNoConvergenceError()
          
# 連続関数クラス
class ContinuousFunction(Vector):
    def __init__(self, x, interval, symbol=None):
        self.interval = interval
        super().__init__(x, symbol=symbol)
    
    def validate(self):
        f = self.value
        evalf = getattr(f, "evalf", None)
        if callable(evalf) == False:
            raise ValidationError("f is not a function.")
        
        # 擬似的なε-δ論法
        step = 1e-2
        epsilon = 1e-2
        delta = epsilon/2
        for x in np.arange(float(self.interval.inf), float(self.interval.sup), step):
            y1 = f.subs(self.symbol, float(x))
            y2 = f.subs(self.symbol, x + delta if x + delta < self.interval.sup else x)
            if y1 == sym.nan or y2 == sym.nan:
                raise DiscontinuousFunctionError(f"f is discontinuas at x={x}")
                
            if abs(y1 - y2) > epsilon:
                raise DiscontinuousFunctionError(f"f is discontinuas at x={x}")
                
    def __call__(self, subs):
        for d in list(subs.values()):
            if self.interval.contains(d) == False:
                raise OutOfIntervalError(f"{d} not in {self.interval}")
            
        return self.value.evalf(subs=subs)
                
    def _operate(self, other, operator):
        operate_func = getattr(self.value, operator, None)
        return self.__class__(operate_func(self._get_primitive_value(other)), self.interval, symbol=self.symbol)

# 連続関数空間クラス
class CSpace(BanachSpace):
    def __init__(self, interval, step):
        super().__init__(lambda f, symbol: ContinuousFunction(f, interval, symbol))
        self.step = step
        self.interval = interval
        self.np_interval = np.arange(interval.inf, interval.sup + step, step)
    
    def norm(self, f, symbol):
        if np.isscalar(f):
            return abs(f)
        return max([abs(f.value.evalf(subs={symbol: x})) for x in self.np_interval])
    
    def distance(self, f, g, symbol):
        return self.norm(f.value-g.value, symbol)
    
    def take(self, *args, symbols=None):
        return super().take(*args, symbols=symbols)

# ヒルベルト空間クラス
class HilbertSpace(BanachSpace, ABC):
    @abstractmethod
    def dot(self, f, g, symbol=None):
        pass
    
# 空集合クラス 
class EmptySet(MathSet): pass
    
# チャプター5用計量ベクトル空間クラス
class CH5_MetricVectorSpace(MathSet):
    def __init__(self, VectorType, ndim, super_space_dim=None, is_ortho_complement=False):
        self.ndim = ndim
        self.VectorType = VectorType
        self.super_space_dim = super_space_dim
        self.is_ortho_complement = is_ortho_complement
        super().__init__(lambda x: VectorType(x, ndim=self._potential_ndim()))
        
    def _potential_ndim(self):
        if self.super_space_dim is None:
            return self.ndim
        
        return self.super_space_dim
    
    def dot(self, x, y):
        if self._potential_ndim() != x.ndim:
            raise NdimMismatchedError(f"n-dim mismatched. x dim should be {self.ndim}")
            
        if self._potential_ndim() != y.ndim:
            raise NdimMismatchedError(f"n-dim mismatched. y dim should be {self.ndim}")
            
        if type(x) != self.VectorType:
            raise ElementTypeMismatchedError(f"type of x should be {self.VectorType.__name__}")
            
        if type(y) != self.VectorType:
            raise ElementTypeMismatchedError(f"type of y should be {self.VectorType.__name__}")
            
        return sum([np.conjugate(pair[0]) * pair[1] for pair in zip(x.value, y.value)])
    
    def norm(self, x):
        return np.real(np.sqrt(self.dot(x, x)))
    
    def distance(self, x, y):
        return self.norm(x - y)
    
    def take(self, *args):
        vecs = super().take(*args)
        if self.super_space_dim is None: return vecs
            
        for vec in [vecs] if type(vecs) == self.VectorType else list(vecs):
            if self.is_ortho_complement:
                for i, elem in enumerate(vec.value):
                    if i >= self.super_space_dim-self.ndim:
                        if elem == 0:
                            raise NdimMismatchedError()
                    else:
                        if elem != 0:
                            raise NdimMismatchedError()
            else:
                for i, elem in enumerate(vec.value):
                    if self.ndim > i:
                        if elem == 0:
                            raise NdimMismatchedError()
                    else:
                        if elem != 0:
                            raise NdimMismatchedError()
        return vecs
    
    def subspace(self, ndim):
        if ndim > self.ndim: raise NdimMismatchedError()
        return CH5_MetricVectorSpace(self.VectorType, ndim, self.ndim)
    
    def ortho_complement(self):
        if self.super_space_dim is None: return EmptySet(lambda _: None)
        
        if self.is_ortho_complement:
            ndim = self.super_space_dim - self.ndim
            return CH5_MetricVectorSpace(self.VectorType, ndim, self.super_space_dim)
        
        ndim = self.super_space_dim - self.ndim
        return CH5_MetricVectorSpace(self.VectorType, ndim, self.super_space_dim, True)

# 区分的に連続で自乗可積分な関数のクラス
class PiecewiseContinuousFunction(Vector):
    def __init__(self, f, interval, symbol, without_validation=False):
        self.interval = interval
        if without_validation:
            self.symbol = symbol
            self.value = f
        else:
            super().__init__(f, symbol=symbol)
    
    def validate(self):
        f = self.value
        result = sym.integrate(f**2, (self.symbol, self.interval.inf, self.interval.sup))
        if result == sym.nan:
            raise NonSquareIntegrableFunctionError(f)
            
    def subs(self, symbol, value):
        return PiecewiseContinuousFunction(self.value.subs(symbol, value), self.interval, symbol)
            
    def __call__(self, subs):
        for d in list(subs.values()):
            if self.interval.contains(d) == False:
                raise OutOfIntervalError(f"{d} not in {self.interval}")
        return self.value.evalf(subs=subs)
                
    def _operate(self, other, operator):
        operate_func = getattr(self.value, operator, None)
        return self.__class__(operate_func(self._get_primitive_value(other)), self.interval, symbol=self.symbol)

# L^2空間クラス
class L2Space(HilbertSpace):
    def __init__(self, interval):
        super().__init__(lambda f, symbol: PiecewiseContinuousFunction(f, interval, symbol))
        self.interval = interval
    
    def dot(self, f, g, symbol):
        return sym.integrate(f.value*sym.conjugate(g.value), (symbol, self.interval.inf, self.interval.sup))
    
    def norm(self, f, symbol):
        return sym.sqrt(sym.integrate(sym.Abs(f.value)**2, (symbol, self.interval.inf, self.interval.sup)))
    
    def distance(self, f, g, symbol):
        return self.norm(f-g, symbol)