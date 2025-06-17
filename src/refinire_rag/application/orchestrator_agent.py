from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from agents import Agent, Runner
from ..models.task import Task, TaskStatus
from ..models.agent import Agent as AgentModel, AgentStatus
from .task_manager import TaskManager
from .agent_manager import AgentManager

class OrchestratorAgent:
    """Orchestrator Agent
    エージェントのオーケストレーションを行うクラス
    """
    def __init__(self, name: str = "Orchestrator"):
        """Initialize OrchestratorAgent
        オーケストレーターエージェントを初期化する

        Args:
            name (str, optional): Agent name. Defaults to "Orchestrator".
        """
        self.name = name
        self.task_manager = TaskManager()
        self.agent_manager = AgentManager()
        self.agent = Agent(
            name=name,
            instructions="You are an orchestrator agent that manages and coordinates other agents to complete tasks."
        )
        self.runner = Runner()

    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process user request
        ユーザーの要求を処理する

        Args:
            request (str): User request

        Returns:
            Dict[str, Any]: Processing result
        """
        # タスクの分析と分解
        tasks = await self._analyze_request(request)
        
        # タスクの実行
        results = await self._execute_tasks(tasks)
        
        return {
            "request": request,
            "tasks": [task.dict() for task in tasks],
            "results": results
        }

    async def _analyze_request(self, request: str) -> List[Task]:
        """Analyze request and create tasks
        要求を分析し、タスクを作成する

        Args:
            request (str): User request

        Returns:
            List[Task]: List of tasks
        """
        # LLMを使用して要求を分析
        analysis = await self.runner.run(
            self.agent,
            f"Analyze the following request and break it down into tasks: {request}"
        )
        
        # 分析結果からタスクを作成
        tasks = []
        for task_info in analysis.tasks:
            task = self.task_manager.create_task(
                task_type=task_info["type"],
                priority=task_info.get("priority", 0),
                parameters=task_info.get("parameters", {})
            )
            tasks.append(task)
            
            # 依存関係の設定
            for dep_id in task_info.get("dependencies", []):
                self.task_manager.add_task_dependency(task.id, dep_id)
        
        return tasks

    async def _execute_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute tasks
        タスクを実行する

        Args:
            tasks (List[Task]): List of tasks to execute

        Returns:
            Dict[str, Any]: Execution results
        """
        results = {}
        
        while tasks:
            # 実行可能なタスクを取得
            ready_tasks = self.task_manager.get_ready_tasks()
            if not ready_tasks:
                break
                
            # 各タスクを実行
            for task in ready_tasks:
                # 適切なエージェントを選択
                agent = self._select_agent_for_task(task)
                if not agent:
                    task.update_status(TaskStatus.FAILED)
                    task.error = "No suitable agent found"
                    continue
                
                # タスクを実行
                try:
                    result = await self._execute_task_with_agent(task, agent)
                    task.update_status(TaskStatus.COMPLETED)
                    task.result = result
                except Exception as e:
                    task.update_status(TaskStatus.FAILED)
                    task.error = str(e)
                
                results[task.id] = {
                    "status": task.status,
                    "result": task.result,
                    "error": task.error
                }
        
        return results

    def _select_agent_for_task(self, task: Task) -> Optional[AgentModel]:
        """Select appropriate agent for task
        タスクに適したエージェントを選択する

        Args:
            task (Task): Task to execute

        Returns:
            Optional[AgentModel]: Selected agent
        """
        # タスクタイプに対応する能力を持つエージェントを探す
        available_agents = self.agent_manager.get_available_agents()
        capable_agents = [agent for agent in available_agents if agent.can_handle_task(task.type)]
        
        if not capable_agents:
            return None
            
        # 最も適したエージェントを選択（例：最も多くの能力を持つエージェント）
        return max(capable_agents, key=lambda a: len(a.capabilities))

    async def _execute_task_with_agent(self, task: Task, agent: AgentModel) -> Any:
        """Execute task with selected agent
        選択したエージェントでタスクを実行する

        Args:
            task (Task): Task to execute
            agent (AgentModel): Agent to execute the task

        Returns:
            Any: Task execution result
        """
        # エージェントの状態を更新
        self.agent_manager.update_agent_status(agent.id, AgentStatus.BUSY)
        agent.current_task_id = task.id
        
        try:
            # タスクを実行
            result = await self.runner.run(
                self.agent,
                f"Execute task {task.id} of type {task.type} with parameters {task.parameters}"
            )
            
            # エージェントの状態を更新
            self.agent_manager.update_agent_status(agent.id, AgentStatus.IDLE)
            agent.current_task_id = None
            
            return result
        except Exception as e:
            # エラー発生時の処理
            self.agent_manager.update_agent_status(agent.id, AgentStatus.ERROR)
            agent.error = str(e)
            raise 