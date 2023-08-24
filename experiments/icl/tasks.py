import torch


class RegressionSequenceDistribution:
    """
    Represents a synthetic in-context regression data set, where each token
    sequence is an i.i.d. sample of inputs and outputs of the form:
        `[ x_1, y_1, x_2, y_2, ..., x_K, y_K ]`
    where `y_i = w . x_i + N(0, noise_variance)` and `w` is sampled from
    `task_distribution`.

    Parameters:

    * `task_distribution : TaskDistribution`
        the distribution of true parameters `w` underlying each sequence.
    * `noise_variance : float >= 0`
        variance for gaussian error added to regression outputs.

    Fields:

    * `task_distribution : TaskDistribution`
        the distribution of true parameters `w` underlying each sequence.
    * `noise_variance : float >= 0`
        variance for gaussian error added to regression outputs.
    """
    def __init__(self, task_distribution, noise_variance=0.25, device='cpu'):
        self.task_distribution = task_distribution
        self.noise_variance = noise_variance
        self.std = noise_variance**0.5


    def get_batch(self, num_examples, batch_size):
        """
        Generate a batch of synthetic data (token sequences) for in-context
        regression.

        Parameters:

        * `num_examples : int >= 0`
            number of in-context examples to generate for each sequence in
            the batch. Note that the number of tokens will be twice as many,
            because each example has one input token followed by one output
            token.
        * `batch_size : int >= 0`
            number of sequences to generate for this batch.
        
        Returns:

        * `xs : tensor(batch_size, num_examples, task_size, device=device)`
            batch of sequences of input vectors.
        * `ys : tensor(batch_size, num_examples, 1, device=device)`
            batch of corresponding sequences of input vectors.
        """
        # shorthands
        B = batch_size
        K = num_examples
        D = self.task_distribution.task_size
        device = self.task_distribution.device

        # sample a batch of random tasks
        ws = self.task_distribution.sample_tasks(B) # -> B D

        # sample i.i.d. inputs and outputs for each task according to the
        # regression model
        xs = torch.normal(
            mean=0.,
            std=1.,
            size=(B, K, D,),
            device=device,
        )
        errors = torch.normal(
            mean=0.,
            std=self.std,
            size=(B, K, 1,),
            device=device,
        )
        ys = xs @ ws.view(B, D, 1) + errors # B K D @ B D . + B K 1 -> B K 1

        return xs, ys


class TaskDistribution:
    """
    Abstract base class: an abstract distribution over regression tasks, each
    task is a vector of size `task_size`, namely the vector of coefficients
    for the regression problem.
    The specific details of the distribution over tasks is to be provided by
    the subclass.

    Constructor parameters:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """
    def __init__(self, task_size, device='cpu'):
        self.task_size = task_size
        self.device = device

    def sample_tasks(self, n):
        """
        produce a sample of `n` tasks from the concrete task distribution

        parameters:

        * `n : int > 0`
            number of tasks to sample.
        
        returns:

        * `tasks : tensor(n, self.task_size, device=self.device)`
            the `n` tasks, each as one row of a 2d tensor.
        """
        return NotImplemented


class DiscreteTaskDistribution(TaskDistribution):
    """
    Represent a fixed finite number of regression tasks `num_tasks` each of
    dimensionality `task_size`
    (the tasks are selected i.i.d. from a `task_size`-dimensional standard
    normal when this instance is constructed).

    Constructor parameters:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `num_tasks : int > 0`
        number of tasks in the discrete set.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `num_tasks : int > 0`
        number of tasks in the discrete set.
    * `tasks : tensor(self.num_tasks, self.task_size, device=self.device)`
        the tasks, each as one row of a 2d tensor.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """
    def __init__(self, task_size, num_tasks, device='cpu'):
        super().__init__(task_size=task_size, device=device)
        self.num_tasks = num_tasks
        self.tasks = torch.normal(
            mean=0.,
            std=1.,
            size=(self.num_tasks, self.task_size),
            device=self.device,
        )


    def sample_tasks(self, n):
        """
        Produce a uniformly random sample (with replacement) of `n` of
        the task distribution's tasks that were generated at construction
        time.

        Parameters:

        * `n : int > 0`
            number of tasks to sample
        
        Returns:

        * `tasks : tensor(n, self.task_size, device=self.device)`
            the `n` tasks, each as one row of a 2d tensor
        """
        task_selection = torch.randint(
            high=self.num_tasks,
            size=(n,),
            device=self.device,
        )
        return self.tasks[task_selection]


class GaussianTaskDistribution(TaskDistribution):
    """
    Represent a gaussian distribution over of all possible regression tasks
    of dimensionality `task_size` (the tasks are selected i.i.d. from a
    `task_size`-dimensional standard normal when this task distribution is
    sampled from).

    Constructor parameters:
    
    * `task_size : int > 0`
        number of dimensions for all tasks
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task_size : int > 0`
        number of dimensions for all tasks
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """


    def sample_tasks(self, n):
        """
        produce a sample of `n` tasks drawn i.i.d. from a standard gaussian
        distribution in `self.task_size` dimensions

        parameters:

        * `n : int > 0`
            number of tasks to sample
        
        returns:

        * `tasks : tensor(n, self.task_size, device=self.device)`
            the `n` tasks, each as one row of a 2d tensor
        """
        tasks = torch.normal(
            mean=0.,
            std=1.,
            size=(n, self.task_size),
            device=self.device,
        )
        return tasks
    

class SingletonTaskDistribution(TaskDistribution):
    """
    An even simpler task distribution, for testing purposes. Just a single
    regression task with a fixed solution vector added at construction time.
    
    Constructor parameters:
    
    * `task : tensor(task_size)`
        fixed single task
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task : tensor(task_size)` (array-like ok)
        fixed single task.
    * `task_size : int > 0`
        number of dimensions for all tasks
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """
    def __init__(self, task, device='cpu'):
        self.task = torch.asarray(task, device=device)
        task_size, = self.task.shape
        super().__init__(task_size=task_size, device=device)


    def sample_tasks(self, n, device='cpu'):
        """
        Produce a batch of the singleton task repeated `n` times in a 2d
        tensor.

        Parameters:

        * `n : int > 0`
            number of tasks to sample
        
        Returns:

        * `tasks : tensor(n, self.task_size, device=device)`
            the task repeated `n` times, each as one row of a 2d tensor
        """
        return self.task.expand(n, -1)


