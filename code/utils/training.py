import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from .regression_methods import _Gradient
from typing import TYPE_CHECKING
from sklearn.model_selection import train_test_split
from sklearn import linear_model

if TYPE_CHECKING:
    from .step_methods import _StepMethod

# Util function for getting points of test-sampling
def get_sample_points(iterations: int, samples: int, logarithmic=True):
    if logarithmic:
        sample_points = np.logspace(0, np.log10(iterations), num=samples, dtype=int)
    else:
        sample_points = np.linspace(0, iterations, num=samples, dtype=int)
    
    return np.unique(sample_points)

# Template for training methods, like gradient descent, and stochastic gradient descent
class _TrainingMethod:
    parameters: npt.NDArray[np.floating]
    def __init__(
        self,
        X: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        gradient: _Gradient,
        starting_parameters: npt.NDArray[np.floating],
        step_method: "_StepMethod",
        t1: int = 100,
    ) -> None:
        self.parameters = starting_parameters.copy()
        self.feature_amount = X.shape[1]
        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.3,random_state=42)
        self.gradient = gradient
        self.step_method = step_method
        self.step_method.setup(self.feature_amount)

        self.step_method.caller = self

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.X_test = self.scaler.transform(self.X_test)
        self.y_mean = self.y.mean()

        self.setup()

    def setup(self):
        ...
    
    def predict(self, X: npt.NDArray[np.floating], already_scaled: bool = False) -> npt.NDArray[np.floating]:
        if not already_scaled:
            X = self.scaler.transform(X)
        
        return X @ self.parameters + self.y_mean
    
    def OLS_Hessian(self):
        return (2.0/self.X.shape[0])*np.transpose(self.X) @ self.X 
    
    def Ridge_Hessian(self,lambda_: float):
        return (2.0/self.X.shape[0])*np.transpose(self.X) @ self.X + 2*lambda_*np.eye(self.X.shape[1])
        
    def One_minus_R2(self):
        y_pred = self.predict(self.X_test, already_scaled=True)
        return mean_squared_error(self.y_test, y_pred)/np.mean((self.y_test - self.y_test.mean())**2)
    
    def analytical_OLS_1_R2(self): 
        X_transpose = np.transpose(self.X)
        parameters = np.linalg.pinv(X_transpose @ self.X) @ X_transpose @ (self.y - self.y_mean)
        y_pred = self.X_test @ parameters + self.y_mean
        return mean_squared_error(self.y_test,y_pred)/np.mean((self.y_test - self.y_test.mean())**2)
    
    def analytical_Ridge_1_R2(self,lambda_: float): 
        X_transpose = np.transpose(self.X)
        parameters = np.linalg.pinv(X_transpose @ self.X + len(self.y)*lambda_*np.eye(self.X.shape[1])) @ X_transpose @ (self.y - self.y_mean)
        y_pred = self.X_test @ parameters + self.y_mean
        return mean_squared_error(self.y_test,y_pred)/np.mean((self.y_test - self.y_test.mean())**2)
    
    def sklearn_lasso_1_R2(self,lambda_: float): 
        reg_lasso = linear_model.Lasso(0.5*lambda_,fit_intercept=True)
        reg_lasso.fit(self.X,self.y-self.y_mean)
        y_pred = reg_lasso.predict(self.X_test) + self.y_mean
        return mean_squared_error(self.y_test,y_pred)/np.mean((self.y_test - self.y_test.mean())**2)
    
    def train(self, *args, **kwargs) -> tuple[npt.ArrayLike, npt.ArrayLike] | None:
        ...


# ========== Training methods ==========

class GradientDescent(_TrainingMethod):
    def train(self, iterations: int = 1000, test_samples: int = 100) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        # MSE sampling
        sample_points = get_sample_points(iterations, test_samples)
        mse_values = np.zeros((len(sample_points)+1, 2))
        mse_values[0] = (1, self.One_minus_R2())
        samples_done = 0
        
        for i in range(iterations):
            gradient = self.gradient(self.X, self.y - self.y_mean, self.parameters)
            self.step_method.training_step(gradient)
            if i + 1 == sample_points[samples_done]:
                mse_values[samples_done + 1] = (i + 2, self.One_minus_R2())
                samples_done += 1
                if samples_done == len(sample_points):
                    break
                
            
    
        return mse_values[:, 0],mse_values[:, 1]
                

class StochasticGradientDescent(_TrainingMethod): 
    def learning_schedule(self,t,t0,t1): 
        return t0/(t + t1)

    def train(self, epochs: int = 1000, n_batches: int = 5, test_samples: int = 1000, logarithmic_sampling = True) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        # MSE sampling
        sample_points = get_sample_points(epochs, test_samples, logarithmic_sampling)
        mse_values = np.zeros((len(sample_points)+1, 2))
        mse_values[0] = (1, self.One_minus_R2())
        samples_done = 0

        n_datapoints = self.X.shape[0]
        batch_size = int(n_datapoints/n_batches)
        initial_learning_rate = self.step_method.learning_rate
        
        for i in range(epochs):
            shuffled_data = np.array(range(n_datapoints))
            np.random.shuffle(shuffled_data)
            for j in range(n_batches): 
                gradient = self.gradient(self.X[shuffled_data][(batch_size*j):(batch_size*(j+1))], self.y[shuffled_data][(batch_size*j):(batch_size*(j+1))] - self.y_mean, self.parameters)
                t = i*n_batches + j
                self.step_method.learning_rate = self.learning_schedule(t,initial_learning_rate*70*n_batches,70*n_batches)
                self.step_method.training_step(gradient)
            
            if i + 1 == sample_points[samples_done]:
                mse_values[samples_done + 1] = (i + 2, self.One_minus_R2())
                samples_done += 1
                if samples_done == len(sample_points):
                    break
                
        return mse_values[:, 0], mse_values[:, 1]