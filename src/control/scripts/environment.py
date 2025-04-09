import numpy as np
import gymnasium as gym
import threading
import multiprocessing as mp
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
    
    @abstractmethod
    def reset(self):
        """
        Reset all environments and return observations.
        
        Returns:
            numpy.ndarray: Observations
        """
        pass
    
    @abstractmethod
    def step_async(self, actions):
        """
        Asynchronously step in all environments.
        
        Args:
            actions: Actions to take in each environment
        """
        pass
    
    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        
        Returns:
            tuple: (observations, rewards, dones, infos)
        """
        pass
    
    def step(self, actions):
        """
        Step in all environments.
        
        Args:
            actions: Actions to take in each environment
            
        Returns:
            tuple: (observations, rewards, dones, infos)
        """
        self.step_async(actions)
        return self.step_wait()
    
    @abstractmethod
    def close(self):
        """
        Close all environments.
        """
        pass
    
    def seed(self, seed=None):
        """
        Set random seeds for all environments.
        
        Args:
            seed: Initial seed
            
        Returns:
            list: List of seeds
        """
        pass


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle).
    """
    def __init__(self, var):
        self.var = var
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.var)
    
    def __setstate__(self, ob):
        import pickle
        self.var = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker function for a subprocess.
    
    Args:
        remote: Process connection for communication with the main process
        parent_remote: Other end of the pipe, to be closed
        env_fn_wrapper: CloudpickleWrapper containing the environment creation function
    """
    parent_remote.close()
    env = env_fn_wrapper.var()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                # Gymnasium uses tuple with truncated flag
                result = env.step(data)
                if len(result) == 5:  # Gymnasium format: obs, reward, terminated, truncated, info
                    ob, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:  # Old gym format: obs, reward, done, info
                    ob, reward, done, info = result
                
                if done:
                    # Gymnasium reset returns (obs, info)
                    try:
                        result = env.reset()
                        if isinstance(result, tuple):  # Gymnasium reset
                            ob, _ = result
                        else:  # Old gym reset
                            ob = result
                    except TypeError:
                        # Fallback for older gym versions
                        ob = env.reset()
                
                remote.send((ob, reward, done, info))
            
            elif cmd == 'reset':
                # Gymnasium reset returns (obs, info)
                try:
                    result = env.reset()
                    if isinstance(result, tuple):  # Gymnasium reset
                        ob, _ = result
                    else:  # Old gym reset
                        ob = result
                except TypeError:
                    # Fallback for older gym versions
                    ob = env.reset()
                
                remote.send(ob)
            
            elif cmd == 'close':
                remote.close()
                break
            
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            
            elif cmd == 'seed':
                # Handle different seeding methods between gym versions
                try:
                    # Modern Gymnasium way - seed in reset
                    _ = env.reset(seed=data)
                    remote.send([data])  # Just return the seed we used
                except (TypeError, ValueError):
                    try:
                        # Older gym way - explicit seed method
                        res = env.seed(data)
                        remote.send(res if res is not None else [data])
                    except (AttributeError, TypeError):
                        # No seeding capability
                        remote.send([data if data is not None else 0])
            
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in subprocesses.
    """
    def __init__(self, env_fns, context=None):
        """
        Initialize a SubprocVecEnv.
        
        Args:
            env_fns: List of functions that create environments
            context: Multiprocessing context
        """
        self.waiting = False
        self.closed = False
        
        if context is None:
            context = mp.get_context()
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])
        self.processes = []
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        
    def step_async(self, actions):
        """
        Asynchronously step in all environments.
        
        Args:
            actions: Actions to take in each environment
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        
        Returns:
            tuple: (observations, rewards, dones, infos)
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        # Transpose the results
        obs, rews, dones, infos = zip(*results)
        
        # Make sure observations are properly formatted as numpy arrays
        if isinstance(obs[0], np.ndarray):
            obs = np.stack(obs)
        else:
            # Handle other observation types (e.g., dict observations)
            pass
            
        return obs, np.stack(rews), np.stack(dones), infos
    
    def reset(self):
        """
        Reset all environments and return observations.
        
        Returns:
            numpy.ndarray: Observations
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)
    
    def close(self):
        """
        Close all environments.
        """
        if self.closed:
            return
        
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()
            
        self.closed = True
        
    def seed(self, seed=None):
        """
        Set random seeds for all environments.
        
        Args:
            seed: Initial seed
            
        Returns:
            list: List of seeds
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
            
        for remote, s in zip(self.remotes, seed):
            remote.send(('seed', s))
            
        return [remote.recv() for remote in self.remotes]


class DummyVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments sequentially in a single process.
    """
    def __init__(self, env_fns):
        """
        Initialize a DummyVecEnv.
        
        Args:
            env_fns: List of functions that create environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        
        self.actions = None
        self.closed = False
        
    def step_async(self, actions):
        """
        Asynchronously step in all environments.
        
        Args:
            actions: Actions to take in each environment
        """
        self.actions = actions
        
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        
        Returns:
            tuple: (observations, rewards, dones, infos)
        """
        results = []
        for env, a in zip(self.envs, self.actions):
            result = env.step(a)
            if len(result) == 5:  # Gymnasium format: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                results.append((obs, reward, done, info))
            else:  # Old gym format: obs, reward, done, info
                results.append(result)
        
        obs, rews, dones, infos = zip(*results)
        
        # Handle episode resets
        new_obs = list(obs)  # Create a mutable copy
        for i, done in enumerate(dones):
            if done:
                # Gymnasium reset returns (obs, info)
                try:
                    result = self.envs[i].reset()
                    if isinstance(result, tuple):  # Gymnasium reset
                        new_obs[i] = result[0]  # Extract observation
                    else:  # Old gym reset
                        new_obs[i] = result
                except TypeError:
                    # Fallback for older gym versions
                    new_obs[i] = self.envs[i].reset()
        
        # Make sure observations are properly formatted as numpy arrays
        if isinstance(new_obs[0], np.ndarray):
            obs_array = np.stack(new_obs)
        else:
            # Handle other observation types (e.g., dict observations)
            obs_array = new_obs
                
        return obs_array, np.stack(rews), np.stack(dones), infos
    
    def reset(self):
        """
        Reset all environments and return observations.
        
        Returns:
            numpy.ndarray: Observations
        """
        obs = []
        for env in self.envs:
            # Gymnasium reset returns (obs, info)
            try:
                result = env.reset()
                if isinstance(result, tuple):  # Gymnasium reset
                    obs.append(result[0])  # Extract observation
                else:  # Old gym reset
                    obs.append(result)
            except TypeError:
                # Fallback for older gym versions
                obs.append(env.reset())
        
        # Make sure observations are properly formatted as numpy arrays
        if isinstance(obs[0], np.ndarray):
            return np.stack(obs)
        else:
            # Handle other observation types
            return obs
    
    def close(self):
        """
        Close all environments.
        """
        if self.closed:
            return
            
        for env in self.envs:
            env.close()
            
        self.closed = True
        
    def seed(self, seed=None):
        """
        Set random seeds for all environments.
        
        Args:
            seed: Initial seed
            
        Returns:
            list: List of seeds
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        
        seeds = []
        for i, (env, s) in enumerate(zip(self.envs, seed)):
            # Try different approaches to setting seed
            try:
                # Modern Gymnasium way - seed in reset
                _ = env.reset(seed=s)
                seeds.append([s])  # Just return the seed we used
            except (TypeError, ValueError):
                try:
                    # Older gym way - explicit seed method
                    result = env.seed(s)
                    seeds.append(result if result is not None else [s])
                except (AttributeError, TypeError):
                    # No seeding capability
                    seeds.append([s if s is not None else 0])
        
        return seeds


def make_vec_env(env_id=None, env_fn=None, num_envs=4, seed=None, track_id=0, render_mode=None):
    """
    Create a vectorized environment.
    
    Args:
        env_id: Gym environment ID
        env_fn: Function to create environment
        num_envs: Number of environments
        seed: Random seed
        track_id: Track ID for NavigationGoal environment
        render_mode: Rendering mode
        
    Returns:
        VecEnv: Vectorized environment
    """
    if env_fn is None:
        if env_id == 'gym_navigation:NavigationGoal-v0':
            # Create gym-navigation environment by Nick Geramanis
            def _make_env():
                env = gym.make('gym_navigation:NavigationGoal-v0', 
                               render_mode=render_mode if render_mode else None,
                               track_id=track_id)
                return env
            env_fn = _make_env
        else:
            # Standard gym environment
            def _make_env():
                try:
                    env = gym.make(env_id, render_mode=render_mode)
                except TypeError:
                    # Fallback for environments that don't support render_mode
                    env = gym.make(env_id)
                return env
            env_fn = _make_env
    
    # Create multiple environments
    env_fns = [env_fn for _ in range(num_envs)]
    
    # Check available CPU count for optimal performance
    cpu_count = mp.cpu_count()
    if num_envs <= 1 or cpu_count <= 1:
        # Use sequential environment if only one env or one CPU
        vec_env = DummyVecEnv(env_fns)
    else:
        # Use parallel environment otherwise
        vec_env = SubprocVecEnv(env_fns)
    
    # Set seeds if provided
    if seed is not None:
        vec_env.seed(seed)
    
    return vec_env