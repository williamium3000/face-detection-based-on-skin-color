import numpy as np
import random
import json

def param_class(parameters, type, evaluation_func):
    '''
        This is a decorator for the parameter class that offers evaluation for the optimizer, this decorator specifies the parameters to be tuned including
        their names, upper and lower bounds and the evaluation method provided by this class.
        param:
            parameters(dict): every parameters to be tuned need to be specifies here
                key(str): specifies the name of the parameter tuned
                value(list): specifies the upper and lower bound of the parameter 
            evaluation_func(str): the name of the method that offers evaluation
    '''
    def wrapper(cls):
        if not isinstance(parameters, dict):
            raise TypeError
        if not isinstance(evaluation_func, str):
            raise TypeError
        if not isinstance(type, dict):
            raise TypeError
        assert set(type.keys()) == set(parameters.keys()), "incorresponding parameters and types"
        for bounds in parameters.values():
            assert bounds[0] <= bounds[1], "lower bound is larger than upper bound"
        def evaluation(self, parameters):
            if not isinstance(parameters, dict):
                raise TypeError
            return getattr(cls, evaluation_func)(self, parameters)
        cls.type = type
        cls.parameters = parameters
        cls.evaluation = evaluation
        return cls
    return wrapper


class APSO_optimizer:
    def __init__(self, population, w, c1, c2, evaluatin_instance):
        self.population = population
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.evaluatin_instance = evaluatin_instance
        self.dimension = len(self.evaluatin_instance.parameters.keys())
 
    def eval(self, swarm):
        param = {}
        for index, key in enumerate(self.evaluatin_instance.parameters.keys()):
            if self.evaluatin_instance.type[key] == "bool":
                param[key] = (swarm[index] > 0.5)
                continue
            param[key] = swarm[index]
        return 1 - self.evaluatin_instance.evaluation(param)

    def _initialization(self):
        self.v = np.zeros((self.population, self.dimension))
        self.swarm = np.zeros((self.population, self.dimension))


        

        # for index, value in enumerate(self.evaluatin_instance.parameters.values()):
        #     self.v[:, index] = 0.5 * np.random.uniform(low = value[0], high = value[1], size = self.population)
        #     self.swarm[:, index] = np.random.uniform(low = value[0], high = value[1], size = self.population)
        bounding_matrix = np.array(list(self.evaluatin_instance.parameters.values()))
        self.v = 0.5 * np.array([(bounding_matrix - bounding_matrix[:, 0].reshape(-1, 1))[:, 1] * np.random.uniform(low = 0, high = 1, size = self.dimension) + bounding_matrix[:, 0] for i in range(self.population)])
        self.swarm = np.array([(bounding_matrix - bounding_matrix[:, 0].reshape(-1, 1))[:, 1] * np.random.uniform(low = 0, high = 1, size = self.dimension) + bounding_matrix[:, 0] for i in range(self.population)])

        self.lower_bound_x = np.array([bounding_matrix[:, 0] for i in range(self.population)])
        self.upper_bound_x = np.array([bounding_matrix[:, 1] for i in range(self.population)])
        self.lower_bound_v = 0.5 * np.array([bounding_matrix[:, 0] for i in range(self.population)])
        self.upper_bound_v = 0.5 * np.array([bounding_matrix[:, 1] for i in range(self.population)])

        self.PbestPos = np.zeros((self.population, self.dimension))
        self.PbestValue = np.zeros(self.population)
        self.GbestValue = 0
        self.GbestPos = np.zeros(self.dimension)



        for i in range(self.swarm.shape[0]):
            self.PbestValue[i] = self.eval(self.swarm[i])

        self.PbestPos = self.swarm.copy()
        index = np.argmin(self.PbestValue)
        self.global_best_particle_index = index
        self.global_best_particle_eval = self.PbestValue[index]
        self.global_worst_particle_index = np.argmax(self.PbestValue)
        self.GbestValue = self.PbestValue[index]
        self.GbestPos = self.PbestPos[index].copy()

    def _ESE_get_f(self):
        dis = np.zeros(self.population)
        for i in range(self.population):
            dis[i] = np.sum((np.sum((self.swarm - self.swarm[i,:]) ** 2, axis = 1) ** 0.5)) / (self.population - 1)
        max = np.max(dis)
        min = np.min(dis)
        self.f = (dis[self.global_best_particle_index] - min) / (max - min)

    def _ESE_param_tuning(self):
        self.w = 1 / (1 + 1.5 * np.exp(-2.6 * self.f))
        eta1 = np.random.uniform(0.05, 0.1)
        eta2 = np.random.uniform(0.05, 0.1)
        c1_t = self.c1
        c2_t = self.c2
        if self.f < 0.25:
            c1_t += eta1
            c2_t += eta2
        elif self.f < 0.5:
            c1_t += eta1
            c2_t -= eta2
        elif self.f < 0.75:
            c1_t += eta1
            c2_t -= eta2
        else:
            c1_t -= eta1
            c2_t += eta2
        if c1_t + c2_t > 4:
            c1_t = 4 * c1_t / (c1_t + c2_t)
            c2_t = 4 * c2_t / (c1_t + c2_t)
        self.c1 = c1_t
        self.c2 = c2_t

    def _ELS(self, current_iteration, total_iteration):
        selected_d = np.random.randint(0, self.dimension - 1)
        sigma = 1 - (1 - 0.1) * current_iteration / total_iteration
        new = self.swarm[self.global_best_particle_index].copy()
        temp = new[selected_d] + (self.upper_bound_x[0][selected_d] - self.lower_bound_x[0][selected_d]) * np.random.normal(loc = 0, scale = sigma) 
        if temp > self.upper_bound_x[0][selected_d]:
            temp = self.upper_bound_x[0][selected_d]
        if (temp < self.lower_bound_x[0][selected_d]):
            temp = self.lower_bound_x[0][selected_d]
        new[selected_d] = temp
        new_eval = self.eval(new)
        if new_eval < self.global_best_particle_eval:
            self.swarm[self.global_best_particle_index] = new
        else:
            self.swarm[self.global_worst_particle_index] = new
       
    def _formalize_v(self):
        self.v[self.v < self.lower_bound_v] = self.lower_bound_v[self.v < self.lower_bound_v]
        self.v[self.v > self.upper_bound_v] = self.upper_bound_v[self.v > self.upper_bound_v]
    def _formalize_x(self):
        self.swarm[self.swarm < self.lower_bound_x] = self.lower_bound_x[self.swarm < self.lower_bound_x]
        self.swarm[self.swarm > self.upper_bound_x] = self.upper_bound_x[self.swarm > self.upper_bound_x]
    def _move(self):
        self.v = self.w * self.v + self.c1 * random.random() * (self.PbestPos - self.swarm) + self.c2 * random.random() * (self.GbestPos - self.swarm)
        self._formalize_v()

        self.swarm += self.v
        self._formalize_x()
    
    def _evaluate(self):
        temp = 10e8
        t_i = -1
        temp_worst = -10e8
        t_i_worst = -1
        eval_temp = -1
        for i in range(self.swarm.shape[0]):
            eval_temp = self.eval(self.swarm[i])
            if self.PbestValue[i] > eval_temp:
                self.PbestValue[i] = eval_temp
                self.PbestPos[i] = self.swarm[i].copy()
            if temp > eval_temp:
                temp = eval_temp
                t_i = i
            if eval_temp > temp_worst:
                temp_worst = eval_temp
                t_i_worst = i
        self.global_best_particle_index = t_i
        self.global_best_particle_eval = temp
        self.global_worst_particle_index = t_i_worst
        if self.GbestValue > temp:
            self.GbestValue = temp
            self.GbestPos = self.PbestPos[t_i].copy()
    
    def setAttr(self, w, c1, c2):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self._initialization()
    
    def fit(self, iteration, internal_iteration, shown = False):
        self._initialization()
        i = 0
        for i in range(iteration):
            if shown:
                    print("current global best value: ", 1 - self.GbestValue)
                    print("current swarm best value: ", 1 - self.PbestValue)
            self._ESE_get_f()
            self._ESE_param_tuning()
            self._ELS(i, iteration)
            for j in range(internal_iteration):
                self._move()
                self._evaluate()
        print("results:", 1 - self.GbestValue)
        print("parameters are :", self.GbestPos)
        return 1 - self.GbestValue, self.GbestPos


if __name__ == "__main__":
    pass
        

        
        
        