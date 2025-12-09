"""
Enhanced Evaluation Script for Agent Context/Memory Management

This script provides comprehensive evaluation of the agent system including:
- Memory system performance (STM/LTM recall accuracy)
- Response quality using LLM-as-judge
- Tool call efficiency and redundancy detection
- Multi-session context management

Usage:
    python enhanced_evaluator.py --agent-url http://localhost:7000 --output results.json
"""

import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
from collections import defaultdict


class EnhancedEvaluator:
    """Comprehensive evaluator for agent memory and performance"""

    def __init__(self, benchmark_file: str, agent_interface, use_llm_judge: bool = False):
        """
        Initialize evaluator

        Args:
            benchmark_file: Path to benchmark.json
            agent_interface: Interface to communicate with the agent
            use_llm_judge: Whether to use LLM-as-judge for quality scoring (requires API key)
        """
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            self.benchmark = json.load(f)

        self.agent = agent_interface
        self.use_llm_judge = use_llm_judge

        self.results = {
            'evaluation_date': datetime.now().isoformat(),
            'benchmark_info': self.benchmark['benchmark_info'],
            'evaluation_config': {
                'use_llm_judge': use_llm_judge,
                'evaluator_version': '2.0'
            },
            'user_results': [],
            'overall_metrics': {},
            'detailed_analysis': {}
        }

        # Track conversation context for memory evaluation
        self.user_contexts = defaultdict(dict)  # user_id -> {session_id -> context}

    def evaluate_all_users(self):
        """Run evaluation for all users in benchmark"""
        print("=" * 80)
        print("ENHANCED EVALUATION - Agent Memory & Performance Analysis")
        print("=" * 80)
        print(f"LLM Judge: {'Enabled' if self.use_llm_judge else 'Disabled (keyword-based)'}")
        print("=" * 80)

        for user in self.benchmark['users']:
            print(f"\n[EVAL] Evaluating {user['user_id']}...")
            user_result = self.evaluate_user(user)
            self.results['user_results'].append(user_result)

        # Calculate overall metrics
        self.calculate_overall_metrics()

        # Generate detailed analysis
        self.generate_detailed_analysis()

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)

        return self.results

    def evaluate_user(self, user: Dict) -> Dict:
        """Evaluate all sessions for a single user"""
        user_result = {
            'user_id': user['user_id'],
            'profile': user['profile'],
            'sessions': [],
            'user_metrics': {},
            'memory_analysis': {
                'stm_performance': {},
                'ltm_performance': {},
                'inter_session_recall': []
            }
        }

        all_responses = []
        all_tool_calls = []
        all_quality_scores = []
        memory_failures = []
        context_losses = []

        for session in user['sessions']:
            print(f"  Session {session['session_id']} ({session['session_info']['context_length']})")

            session_result = self.evaluate_session(user['user_id'], session, user['profile'])
            user_result['sessions'].append(session_result)

            # Aggregate data
            all_responses.extend(session_result['responses'])
            all_tool_calls.extend(session_result['tool_calls_log'])
            all_quality_scores.extend([r['quality_score'] for r in session_result['responses'] if r['quality_score'] > 0])
            memory_failures.extend(session_result['memory_failures'])
            context_losses.extend(session_result['context_losses'])

        # Calculate user-level metrics
        user_result['user_metrics'] = {
            'total_requests': len(all_responses),
            'task_completion_rate': self.calculate_completion_rate(all_responses),
            'average_quality_score': sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0,
            'total_memory_failures': len(memory_failures),
            'total_context_losses': len(context_losses),
            'total_tool_calls': len(all_tool_calls),
            'memory_failure_rate': len(memory_failures) / len(all_responses) if all_responses else 0,
            'average_tools_per_request': len(all_tool_calls) / len(all_responses) if all_responses else 0
        }

        # Analyze memory performance
        user_result['memory_analysis'] = self.analyze_user_memory(user['user_id'], user_result['sessions'])

        return user_result

    def evaluate_session(self, user_id: str, session: Dict, user_profile: Dict) -> Dict:
        """Evaluate a single session"""
        session_result = {
            'session_id': session['session_id'],
            'context_length': session['session_info']['context_length'],
            'tools_required': session['session_info']['tools_required'],
            'responses': [],
            'tool_calls_log': [],
            'memory_failures': [],
            'context_losses': [],
            'tool_efficiency_analysis': {}
        }

        # Start new session
        self.agent.start_new_session(user_id, session['session_id'])

        conversation_history = []
        session_knowledge = {}  # Track facts established in this session

        for request in session['requests']:
            turn_num = request['turn']
            user_request = request['request']

            print(f"    Turn {turn_num}: {user_request[:60]}...")

            # Query agent
            response_data = self.agent.query(
                user_request=user_request,
                conversation_history=conversation_history
            )

            # Extract response and tool calls
            agent_response = response_data.get('response', '')
            tool_calls = response_data.get('tool_calls', [])

            # Update conversation history
            conversation_history.append({
                'turn': turn_num,
                'user': user_request,
                'agent': agent_response
            })

            # Extract knowledge from this turn
            turn_knowledge = self.extract_knowledge(user_request, agent_response, tool_calls)
            session_knowledge.update(turn_knowledge)

            # Evaluate this turn
            turn_eval = self.evaluate_turn(
                turn_num=turn_num,
                user_request=user_request,
                agent_response=agent_response,
                tool_calls=tool_calls,
                conversation_history=conversation_history,
                session_info=session,
                session_knowledge=session_knowledge,
                user_profile=user_profile
            )

            session_result['responses'].append(turn_eval)
            session_result['tool_calls_log'].extend(tool_calls)

            # Check for memory failures
            if turn_eval['memory_failure']:
                session_result['memory_failures'].append({
                    'turn': turn_num,
                    'request': user_request,
                    'failure_type': turn_eval['memory_failure_type'],
                    'description': turn_eval['memory_failure_description']
                })

            # Check for context loss
            if turn_eval['context_loss']:
                session_result['context_losses'].append({
                    'turn': turn_num,
                    'request': user_request,
                    'description': turn_eval['context_loss_description']
                })

        # Analyze tool efficiency for this session
        session_result['tool_efficiency_analysis'] = self.analyze_tool_efficiency(
            session_result['tool_calls_log'],
            conversation_history
        )

        # Store session context for inter-session analysis
        self.user_contexts[user_id][session['session_id']] = {
            'knowledge': session_knowledge,
            'conversation': conversation_history,
            'profile_updates': self.extract_profile_updates(conversation_history, user_profile)
        }

        return session_result

    def evaluate_turn(self, turn_num: int, user_request: str, agent_response: str,
                     tool_calls: List[Dict], conversation_history: List[Dict],
                     session_info: Dict, session_knowledge: Dict, user_profile: Dict) -> Dict:
        """Evaluate a single turn with comprehensive metrics"""

        turn_eval = {
            'turn': turn_num,
            'user_request': user_request,
            'agent_response': agent_response,
            'tool_calls': tool_calls,
            'task_completed': None,
            'quality_score': 0,
            'memory_failure': False,
            'memory_failure_type': None,
            'memory_failure_description': None,
            'context_loss': False,
            'context_loss_description': None,
            'redundant_tool_calls': [],
            'missing_tool_calls': [],
            'stm_recall_test': None,
            'ltm_recall_test': None,
            'notes': ""
        }

        # Detect memory issues
        memory_issues = self.detect_memory_issues(
            user_request, agent_response, conversation_history,
            session_knowledge, user_profile
        )
        turn_eval.update(memory_issues)

        # Detect tool efficiency issues
        tool_issues = self.detect_tool_issues(
            user_request, agent_response, tool_calls, conversation_history, session_knowledge
        )
        turn_eval['redundant_tool_calls'] = tool_issues['redundant']
        turn_eval['missing_tool_calls'] = tool_issues['missing']

        # Evaluate response quality
        if self.use_llm_judge:
            turn_eval['quality_score'] = self.llm_judge_quality(
                user_request, agent_response, conversation_history
            )
            turn_eval['task_completed'] = turn_eval['quality_score'] >= 3
        else:
            # Keyword-based quality assessment
            quality_assessment = self.keyword_based_quality(user_request, agent_response, tool_calls)
            turn_eval['quality_score'] = quality_assessment['score']
            turn_eval['task_completed'] = quality_assessment['completed']

        return turn_eval

    def extract_knowledge(self, user_request: str, agent_response: str, tool_calls: List[Dict]) -> Dict:
        """Extract factual knowledge established in this turn"""
        knowledge = {}

        # Extract numbers and calculations
        calc_pattern = r'(\d+(?:\.\d+)?)\s*(?:billion|million|thousand|GB|MB|hours|days|parameters)?'
        numbers = re.findall(calc_pattern, agent_response)

        # Extract from tool calls
        for tool_call in tool_calls:
            if tool_call['tool'] == 'calculator':
                result = tool_call.get('result')
                if result:
                    knowledge[f"calculated_{tool_call.get('arguments', {}).get('expression', 'value')}"] = result

        # Extract entities mentioned
        if 'GPT-3' in agent_response or 'GPT-3' in user_request:
            knowledge['discussed_model'] = 'GPT-3'

        return knowledge

    def detect_memory_issues(self, user_request: str, agent_response: str,
                            conversation_history: List[Dict], session_knowledge: Dict,
                            user_profile: Dict) -> Dict:
        """Detect memory failures (STM and LTM)"""
        issues = {
            'memory_failure': False,
            'memory_failure_type': None,
            'memory_failure_description': None,
            'stm_recall_test': None,
            'ltm_recall_test': None
        }

        request_lower = user_request.lower()
        response_lower = agent_response.lower()

        # Reference keywords indicating agent should recall previous info
        reference_keywords = [
            'we calculated', 'we discussed', 'you mentioned', 'you found',
            'last time', 'before', 'earlier', 'previously', 'from before',
            'the size we calculated', 'the model we talked about',
            'remember', 'recall', 'as you said'
        ]

        # Confusion keywords indicating agent doesn't remember
        confusion_keywords = [
            "i don't have", "i don't recall", "could you remind me",
            "what was", "which one", "can you clarify", "i'm not sure what",
            "i don't remember", "please specify", "what do you mean by"
        ]

        # Check if user is referencing past information
        is_reference_request = any(keyword in request_lower for keyword in reference_keywords)

        if is_reference_request:
            # This is an STM recall test
            issues['stm_recall_test'] = True

            # Check if agent shows confusion
            shows_confusion = any(keyword in response_lower for keyword in confusion_keywords)

            if shows_confusion:
                issues['memory_failure'] = True
                issues['memory_failure_type'] = 'stm_failure'
                issues['memory_failure_description'] = (
                    f"Agent failed to recall information from earlier in the session. "
                    f"User referenced previous context but agent showed confusion."
                )

        # Check for inter-session memory (LTM) requirements
        ltm_indicators = [
            'remember i', 'as i mentioned before', 'my interest', 'i told you',
            'you know i', 'i\'m interested in'
        ]

        is_ltm_test = any(indicator in request_lower for indicator in ltm_indicators)

        if is_ltm_test:
            issues['ltm_recall_test'] = True

            # Check if agent acknowledges the profile/preference
            acknowledges = any(keyword in response_lower for keyword in [
                'your interest', 'you mentioned', 'as you said', 'you\'re interested'
            ])

            if not acknowledges and any(keyword in response_lower for keyword in confusion_keywords):
                issues['memory_failure'] = True
                issues['memory_failure_type'] = 'ltm_failure'
                issues['memory_failure_description'] = (
                    f"Agent failed to recall user profile/preferences from previous sessions. "
                    f"This should be stored in long-term memory (Mem0)."
                )

        return issues

    def detect_tool_issues(self, user_request: str, agent_response: str,
                          tool_calls: List[Dict], conversation_history: List[Dict],
                          session_knowledge: Dict) -> Dict:
        """Detect redundant or missing tool calls"""
        issues = {
            'redundant': [],
            'missing': []
        }

        # Check for redundant calculator calls
        if any(tc['tool'] == 'calculator' for tc in tool_calls):
            calc_expression = None
            for tc in tool_calls:
                if tc['tool'] == 'calculator':
                    calc_expression = tc.get('arguments', {}).get('expression')

            # Check if we already calculated this
            if calc_expression and f"calculated_{calc_expression}" in session_knowledge:
                issues['redundant'].append({
                    'tool': 'calculator',
                    'reason': 'Recalculating previously computed value',
                    'expression': calc_expression
                })

        # Check for missing calculator calls
        calc_indicators = ['calculate', 'how many', 'how much', 'total', 'sum', '+', '*', '/']
        needs_calculation = any(indicator in user_request.lower() for indicator in calc_indicators)
        has_calculator_call = any(tc['tool'] == 'calculator' for tc in tool_calls)

        if needs_calculation and not has_calculator_call:
            # Check if agent provided a numeric answer without calculation
            has_numbers = bool(re.search(r'\d+', agent_response))
            if has_numbers:
                issues['missing'].append({
                    'tool': 'calculator',
                    'reason': 'Provided numeric answer without using calculator tool',
                    'request': user_request[:100]
                })

        # Check for redundant search calls
        search_tools = ['search_web_with_content', 'reddit_search']
        search_calls = [tc for tc in tool_calls if tc['tool'] in search_tools]

        if len(search_calls) > 1:
            # Check if searching for same/similar query
            queries = [tc.get('arguments', {}).get('query', '') for tc in search_calls]
            if len(set(queries)) < len(queries):
                issues['redundant'].append({
                    'tool': 'search',
                    'reason': 'Multiple searches with identical queries',
                    'queries': queries
                })

        return issues

    def keyword_based_quality(self, user_request: str, agent_response: str,
                             tool_calls: List[Dict]) -> Dict:
        """Simple keyword-based quality assessment (fallback when no LLM judge)"""
        score = 3  # Default neutral score
        completed = True

        response_lower = agent_response.lower()

        # Positive indicators
        if any(tc.get('result') for tc in tool_calls):
            score += 0.5  # Successfully used tools

        if len(agent_response) > 50:
            score += 0.5  # Substantial response

        # Negative indicators
        error_keywords = ['error', 'failed', 'could not', 'unable to', 'sorry']
        if any(keyword in response_lower for keyword in error_keywords):
            score -= 1
            completed = False

        confusion_keywords = ["i don't have", "i'm not sure", "unclear", "don't understand"]
        if any(keyword in response_lower for keyword in confusion_keywords):
            score -= 0.5

        # Clamp score between 1 and 5
        score = max(1, min(5, score))

        return {'score': score, 'completed': completed}

    def llm_judge_quality(self, user_request: str, agent_response: str,
                         conversation_history: List[Dict]) -> float:
        """Use LLM to judge response quality (1-5 scale)"""
        # TODO: Implement LLM-as-judge using OpenAI API
        # This would send the request and response to GPT-4 with a judging prompt
        # For now, return keyword-based fallback
        return self.keyword_based_quality(user_request, agent_response, [])['score']

    def analyze_tool_efficiency(self, tool_calls: List[Dict],
                               conversation_history: List[Dict]) -> Dict:
        """Analyze tool usage efficiency for a session"""
        tool_stats = defaultdict(int)
        for tool_call in tool_calls:
            tool_stats[tool_call['tool']] += 1

        return {
            'total_tool_calls': len(tool_calls),
            'tool_distribution': dict(tool_stats),
            'average_tools_per_turn': len(tool_calls) / len(conversation_history) if conversation_history else 0,
            'unique_tools_used': len(tool_stats)
        }

    def extract_profile_updates(self, conversation_history: List[Dict],
                               user_profile: Dict) -> List[str]:
        """Extract new information about user that should update their profile"""
        updates = []

        for turn in conversation_history:
            user_msg = turn.get('user', '').lower()

            # Check for interest declarations
            if 'interested in' in user_msg or 'i like' in user_msg:
                updates.append(f"User expressed interest: {turn.get('user', '')[:100]}")

            # Check for preference statements
            if 'i prefer' in user_msg or 'i want' in user_msg:
                updates.append(f"User stated preference: {turn.get('user', '')[:100]}")

        return updates

    def analyze_user_memory(self, user_id: str, sessions: List[Dict]) -> Dict:
        """Analyze STM and LTM performance for a user"""
        stm_tests = []
        ltm_tests = []

        for session in sessions:
            for response in session['responses']:
                if response.get('stm_recall_test'):
                    stm_tests.append({
                        'session': session['session_id'],
                        'turn': response['turn'],
                        'success': not response['memory_failure'],
                        'failure_type': response.get('memory_failure_type')
                    })

                if response.get('ltm_recall_test'):
                    ltm_tests.append({
                        'session': session['session_id'],
                        'turn': response['turn'],
                        'success': not response['memory_failure'],
                        'failure_type': response.get('memory_failure_type')
                    })

        return {
            'stm_performance': {
                'total_tests': len(stm_tests),
                'successful_recalls': sum(1 for t in stm_tests if t['success']),
                'recall_rate': sum(1 for t in stm_tests if t['success']) / len(stm_tests) if stm_tests else 0,
                'tests': stm_tests
            },
            'ltm_performance': {
                'total_tests': len(ltm_tests),
                'successful_recalls': sum(1 for t in ltm_tests if t['success']),
                'recall_rate': sum(1 for t in ltm_tests if t['success']) / len(ltm_tests) if ltm_tests else 0,
                'tests': ltm_tests
            }
        }

    def calculate_completion_rate(self, responses: List[Dict]) -> float:
        """Calculate task completion rate from responses"""
        completed = sum(1 for r in responses if r.get('task_completed') == True)
        total = len(responses)

        if total == 0:
            return 0.0

        if all(r.get('task_completed') is None for r in responses):
            return None

        return completed / total

    def calculate_overall_metrics(self):
        """Calculate overall metrics across all users"""
        all_quality_scores = []
        all_memory_failures = 0
        all_context_losses = 0
        all_requests = 0
        all_tool_calls = 0
        all_stm_tests = 0
        all_stm_successes = 0
        all_ltm_tests = 0
        all_ltm_successes = 0

        for user_result in self.results['user_results']:
            user_metrics = user_result['user_metrics']
            all_requests += user_metrics['total_requests']
            all_memory_failures += user_metrics['total_memory_failures']
            all_context_losses += user_metrics.get('total_context_losses', 0)
            all_tool_calls += user_metrics['total_tool_calls']

            # Collect quality scores
            for session in user_result['sessions']:
                for response in session['responses']:
                    if response['quality_score'] > 0:
                        all_quality_scores.append(response['quality_score'])

            # Collect memory performance
            stm_perf = user_result['memory_analysis']['stm_performance']
            ltm_perf = user_result['memory_analysis']['ltm_performance']

            all_stm_tests += stm_perf['total_tests']
            all_stm_successes += stm_perf['successful_recalls']
            all_ltm_tests += ltm_perf['total_tests']
            all_ltm_successes += ltm_perf['successful_recalls']

        self.results['overall_metrics'] = {
            'total_requests': all_requests,
            'total_memory_failures': all_memory_failures,
            'total_context_losses': all_context_losses,
            'memory_failure_rate': all_memory_failures / all_requests if all_requests else 0,
            'average_quality_score': sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0,
            'total_tool_calls': all_tool_calls,
            'average_tool_calls_per_request': all_tool_calls / all_requests if all_requests else 0,
            'stm_recall_rate': all_stm_successes / all_stm_tests if all_stm_tests else 0,
            'ltm_recall_rate': all_ltm_successes / all_ltm_tests if all_ltm_tests else 0,
            'stm_tests_count': all_stm_tests,
            'ltm_tests_count': all_ltm_tests
        }

    def generate_detailed_analysis(self):
        """Generate human-readable analysis of results"""
        overall = self.results['overall_metrics']

        analysis = {
            'summary': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }

        # Summary
        analysis['summary'] = {
            'total_interactions': overall['total_requests'],
            'avg_quality': round(overall['average_quality_score'], 2),
            'memory_health': 'Good' if overall['memory_failure_rate'] < 0.2 else 'Needs Improvement',
            'stm_health': f"{overall['stm_recall_rate']:.1%} recall rate",
            'ltm_health': f"{overall['ltm_recall_rate']:.1%} recall rate"
        }

        # Identify strengths
        if overall['average_quality_score'] >= 4:
            analysis['strengths'].append("High quality responses (avg >= 4.0)")

        if overall['stm_recall_rate'] >= 0.8:
            analysis['strengths'].append(f"Excellent STM recall ({overall['stm_recall_rate']:.1%})")

        if overall['ltm_recall_rate'] >= 0.7:
            analysis['strengths'].append(f"Good LTM recall ({overall['ltm_recall_rate']:.1%})")

        if overall['average_tool_calls_per_request'] <= 2:
            analysis['strengths'].append("Efficient tool usage (avg <= 2 per request)")

        # Identify weaknesses
        if overall['memory_failure_rate'] > 0.3:
            analysis['weaknesses'].append(f"High memory failure rate ({overall['memory_failure_rate']:.1%})")

        if overall['stm_recall_rate'] < 0.6:
            analysis['weaknesses'].append(f"Poor STM recall ({overall['stm_recall_rate']:.1%})")
            analysis['recommendations'].append("Consider increasing Redis STM window size or improving context passing")

        if overall['ltm_recall_rate'] < 0.5:
            analysis['weaknesses'].append(f"Poor LTM recall ({overall['ltm_recall_rate']:.1%})")
            analysis['recommendations'].append("Review Mem0 configuration and memory extraction prompts")

        if overall['average_tool_calls_per_request'] > 3:
            analysis['weaknesses'].append(f"Excessive tool calls ({overall['average_tool_calls_per_request']:.1f} per request)")
            analysis['recommendations'].append("Implement tool call caching or improve agent planning")

        self.results['detailed_analysis'] = analysis

    def save_results(self, output_file: str):
        """Save evaluation results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Results saved to: {output_file}")

    def print_summary(self):
        """Print comprehensive summary of evaluation results"""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        overall = self.results['overall_metrics']
        analysis = self.results['detailed_analysis']

        print(f"\n[OVERALL METRICS]")
        print(f"  Total Requests: {overall['total_requests']}")
        print(f"  Average Quality Score: {overall['average_quality_score']:.2f} / 5.0")
        print(f"  Total Tool Calls: {overall['total_tool_calls']}")
        print(f"  Avg Tools/Request: {overall['average_tool_calls_per_request']:.2f}")

        print(f"\n[MEMORY PERFORMANCE]")
        print(f"  Memory Failure Rate: {overall['memory_failure_rate']:.1%}")
        print(f"  STM (Redis) Recall: {overall['stm_recall_rate']:.1%} ({overall['stm_tests_count']} tests)")
        print(f"  LTM (Mem0) Recall: {overall['ltm_recall_rate']:.1%} ({overall['ltm_tests_count']} tests)")

        print(f"\n[ANALYSIS]")
        print(f"  Overall Health: {analysis['summary']['memory_health']}")

        if analysis['strengths']:
            print(f"\n  ✓ Strengths:")
            for strength in analysis['strengths']:
                print(f"    - {strength}")

        if analysis['weaknesses']:
            print(f"\n  ✗ Weaknesses:")
            for weakness in analysis['weaknesses']:
                print(f"    - {weakness}")

        if analysis['recommendations']:
            print(f"\n  → Recommendations:")
            for rec in analysis['recommendations']:
                print(f"    - {rec}")

        print(f"\n[PER-USER BREAKDOWN]")
        for user_result in self.results['user_results']:
            user_id = user_result['user_id']
            metrics = user_result['user_metrics']
            mem_analysis = user_result['memory_analysis']

            print(f"\n  {user_id}:")
            print(f"    Requests: {metrics['total_requests']}")
            print(f"    Avg Quality: {metrics.get('average_quality_score', 0):.2f}")
            print(f"    Memory Failures: {metrics['total_memory_failures']}")
            print(f"    STM Recall: {mem_analysis['stm_performance']['recall_rate']:.1%}")
            print(f"    LTM Recall: {mem_analysis['ltm_performance']['recall_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced evaluation for agent memory and performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--benchmark',
        default='benchmark.json',
        help='Path to benchmark JSON file'
    )

    parser.add_argument(
        '--agent-url',
        default='http://localhost:7000',
        help='Agent API endpoint URL'
    )

    parser.add_argument(
        '--output',
        default='enhanced_results.json',
        help='Output file for results'
    )

    parser.add_argument(
        '--use-llm-judge',
        action='store_true',
        help='Use LLM-as-judge for quality scoring (requires OpenAI API key)'
    )

    args = parser.parse_args()

    # Import agent interface
    from agent_stack_interface import AgentStackInterface

    print("=" * 80)
    print("ENHANCED AGENT EVALUATION")
    print("=" * 80)
    print(f"Agent URL: {args.agent_url}")
    print(f"Benchmark: {args.benchmark}")
    print(f"LLM Judge: {'Enabled' if args.use_llm_judge else 'Disabled'}")
    print("=" * 80)

    # Initialize agent interface
    agent = AgentStackInterface(base_url=args.agent_url)

    # Test connection
    if not agent.health_check():
        print("\n[WARNING] Agent health check failed!")
        print("Make sure the agent is running:")
        print("  cd ~/agent-stack && docker compose up -d")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Initialize evaluator
    evaluator = EnhancedEvaluator(
        benchmark_file=args.benchmark,
        agent_interface=agent,
        use_llm_judge=args.use_llm_judge
    )

    # Run evaluation
    print("\n[Starting evaluation...]\n")
    results = evaluator.evaluate_all_users()

    # Save results
    evaluator.save_results(args.output)

    # Print summary
    evaluator.print_summary()

    print("\n" + "=" * 80)
    print("[NEXT STEPS]")
    print("=" * 80)
    print(f"1. Review results: cat {args.output}")
    print(f"2. Generate visualization: python generate_report.py --input {args.output}")
    print(f"3. Compare with baseline: python compare_results.py baseline_results.json {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()