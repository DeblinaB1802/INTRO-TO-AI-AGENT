import logging
import json
import re
from enum import Enum
from typing import List, Dict
from dotenv import load_dotenv
from call_llm import call_openai
from summarizer import Summarizer

class QuizType(Enum):
    TOPIC = "topic"
    SESSION = "session"
    PAST_TOPICS = "past_topics"

class QuizGenerator:
    def __init__(self):
        self.current_quiz = []
        self.user_answers = []
        self.scores = []
        
    def generate_quiz_questions(self, query: str, quiz_type: QuizType, session_summary: str, past_summary: str) -> List[Dict]:
        """Generate quiz questions based on type and content"""

        if quiz_type == QuizType.TOPIC:
            source = "the specified topic"
            content = query
        elif quiz_type == QuizType.SESSION:
            source = "current session content"
            content = session_summary
        else:
            source = "past studied topics"
            content = past_summary

        prompt = f"""
        Generate a 10-question quiz based on {source}:
        
        Content: {content}
        Query: {query}
        
        Create ONLY descriptive questions (no multiple choice). Questions should require detailed explanations and understanding.
        
        Create questions of varying difficulty:
        - 4 easy questions (basic recall and definitions)
        - 4 medium questions (understanding and application)
        - 2 hard questions (analysis and synthesis)
        
        Format each question as JSON:
        {{
            "question_number": 1,
            "difficulty": "easy/medium/hard",
            "question": "Your descriptive question here?",
            "expected_answer": "Brief outline of what a good answer should include"
        }}
        
        Return ONLY a valid JSON array of 10 questions, no other text.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = call_openai(messages)
        
        try:
            cleaned = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE).strip()
            questions = json.loads(cleaned)
            print(questions)
            return questions
        except json.JSONDecodeError:
            logging.error("Failed to parse quiz questions JSON")
            return self._create_fallback_questions(query)
    
    def _create_fallback_questions(self, query: str) -> List[Dict]:
        """Create fallback questions if JSON parsing fails"""

        return [
            {
                "question_number": 1,
                "difficulty": "easy",
                "question": f"What are the key concepts related to {query}?",
                "expected_answer": "Should cover main ideas and definitions"
            },
            {
                "question_number": 2,
                "difficulty": "medium", 
                "question": f"How would you explain {query} to someone unfamiliar with the topic?",
                "expected_answer": "Should provide clear explanation with examples"
            }
        ]
    
    def evaluate_answer(self, question: Dict, user_answer: str) -> Dict:
        """Evaluate user's answer using LLM"""

        prompt = f"""
        Question: {question['question']}
        Expected Answer Guidelines: {question['expected_answer']}
        Student's Answer: {user_answer}
        Question Difficulty: {question['difficulty']}
        
        Evaluate the student's answer and provide:
        1. A score out of 10
        2. Detailed feedback explaining what was good and what could be improved
        3. Additional insights or corrections if needed
        
        Be constructive and encouraging while being honest about the quality of the answer.
        
        Format your response as JSON output:
        {{
        "score": (a score out of 10)
        "feedback": [Your detailed feedback here]
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = call_openai(messages)
        
        # Parse score from response
        try:
            cleaned = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE).strip()
            eval = json.loads(cleaned)
            score = eval['score']
            feedback = eval['feedback']
        except:
            score = 0
            feedback = 'Your answer could not be evaluated.'
        return {
            "score": score,
            "feedback": feedback
        }
    
    def run_interactive_quiz(self, query: str, quiz_type: QuizType, session_summary: str, past_summary: str):
        """Run the complete interactive quiz"""

        print(f"\n Starting Quiz: {query}")
        print("=" * 50)
        
        # Generate questions
        print("📝 Generating quiz questions...")
        self.current_quiz = self.generate_quiz_questions(query, quiz_type, session_summary, past_summary)
        
        if not self.current_quiz:
            print("❌ Failed to generate quiz questions. Please try again.")
            return
            
        print(f"✅ Generated {len(self.current_quiz)} questions\n")
        
        # Reset tracking variables
        self.user_answers = []
        self.scores = []
        
        # Run quiz loop
        for i, question in enumerate(self.current_quiz, 1):
            print(f"\n📚 Question {i}/{len(self.current_quiz)} ({question['difficulty'].upper()} difficulty)")
            print("-" * 40)
            print(f"❓ {question['question']}")
            print()
            
            # Get user answer
            user_answer = input("Your answer: ").strip()
            
            if not user_answer:
                print("⚠️  Empty answer submitted. Moving to next question...")
                self.user_answers.append("")
                self.scores.append(0)
                continue
            
            print("\n🤔 Evaluating your answer...")
            
            # Evaluate answer
            evaluation = self.evaluate_answer(question, user_answer)
            self.user_answers.append(user_answer)
            self.scores.append(evaluation['score'])
            
            # Show feedback
            print("\n📊 EVALUATION:")
            print(f"Score: {evaluation['score']}")
            print(f"Feedback: {evaluation['feedback']}")
            print("\n" + "="*50)
            
            # Ask if user wants to continue
            if i < len(self.current_quiz):
                continue_quiz = input("\nPress Enter to continue to next question (or 'q' to quit): ").strip().lower()
                if continue_quiz == 'q':
                    print("Quiz ended early.")
                    break
        
        # Show final results
        self._show_final_results()
    
    def _show_final_results(self):
        """Display final quiz results and overall feedback"""

        if not self.scores:
            print("No scores to display.")
            return
            
        total_score = sum(self.scores)
        max_score = len(self.scores) * 10
        percentage = (total_score / max_score) * 100
        
        print("\n" + "="*60)
        print("🏆 FINAL QUIZ RESULTS")
        print("="*60)
        print(f"📊 Total Score: {total_score}/{max_score} ({percentage:.1f}%)")
        print(f"📈 Questions Answered: {len(self.scores)}")
        print(f"📋 Average Score per Question: {total_score/len(self.scores):.1f}/10")
        
        # Performance breakdown
        easy_scores = [self.scores[i] for i, q in enumerate(self.current_quiz[:len(self.scores)]) if q.get('difficulty') == 'easy']
        medium_scores = [self.scores[i] for i, q in enumerate(self.current_quiz[:len(self.scores)]) if q.get('difficulty') == 'medium']
        hard_scores = [self.scores[i] for i, q in enumerate(self.current_quiz[:len(self.scores)]) if q.get('difficulty') == 'hard']
        
        if easy_scores:
            print(f"📗 Easy Questions: {sum(easy_scores)}/{len(easy_scores)*10} (Avg: {sum(easy_scores)/len(easy_scores):.1f}/10)")
        if medium_scores:
            print(f"📙 Medium Questions: {sum(medium_scores)}/{len(medium_scores)*10} (Avg: {sum(medium_scores)/len(medium_scores):.1f}/10)")
        if hard_scores:
            print(f"📕 Hard Questions: {sum(hard_scores)}/{len(hard_scores)*10} (Avg: {sum(hard_scores)/len(hard_scores):.1f}/10)")
        
        # Generate overall feedback
        print("\n🎯 OVERALL FEEDBACK:")
        self._generate_overall_feedback(percentage, easy_scores, medium_scores, hard_scores)
        print("="*60)
    
    def _generate_overall_feedback(self, percentage: float, easy_scores: List, medium_scores: List, hard_scores: List):
        """Generate overall performance feedback using LLM"""
        feedback_prompt = f"""
        Provide encouraging and constructive overall feedback for a student who completed a quiz with:
        - Overall score: {percentage:.1f}%
        - Easy questions average: {sum(easy_scores)/len(easy_scores):.1f}/10 if easy_scores else 'N/A'
        - Medium questions average: {sum(medium_scores)/len(medium_scores):.1f}/10 if medium_scores else 'N/A'
        - Hard questions average: {sum(hard_scores)/len(hard_scores):.1f}/10 if hard_scores else 'N/A'
        
        Give specific advice on:
        1. What they did well
        2. Areas for improvement
        3. Study recommendations
        4. Encouragement for continued learning
        
        Keep it positive and motivating while being honest about performance.
        """
        
        messages = [{"role": "user", "content": feedback_prompt}]
        feedback = call_openai(messages)
        print(feedback)

