from rag import prep_ragdb, retrieve_chunks
import openai
import logging
from ethical_guardrail import check_ethical_compliance
from chat_history import add_to_history, load_chat_history, get_optimized_context, extract_entities, get_entity_context, summarize_history
from classify_query import classify_query, structure_query
from tools import search_tavily, search_wikipedia
from reasoning import plan_execute_refine_math, fallback_strategy, self_correct_response, evaluate_confidence
from call_llm import call_openai
from prompt_design import embed_domains, build_domain_specific_prompt, get_style_conditioned_prompt

api_key = "your_api_key"
openai.api_key = api_key

def main():
    print("Welcome to Study Buddy — your partner in learning. Let’s explore knowledge together!")
    print("\n\nJust a moment! Your Study Buddy is getting everything ready.")

    vectordb = prep_ragdb()
    domain_embeds = embed_domains()

    load_chat_history()

    print("\n\nAll set! Your Study Buddy is now ready to assist you.")

    user_level = input("\nEnter your educational details: (choose from ['middle_school', 'high_school', 'college', 'postgraduate', 'expert'] :")
    style = get_style_conditioned_prompt(user_level)

    while True:

        question = input("\nEnter your question (or type 'exit' to quit):")
        if question.lower() == "exit":
            print("Good Bye!!!")
            break

        is_complaint, warning = check_ethical_compliance(question, True)
        if not is_complaint:
            print(warning)
            continue

        add_to_history(role="user", content=question)
        extract_entities(question)
        context = get_optimized_context(question, max_tokens=5000)
        if not context:
            context = [{"role": "user/Assistant", "content": summarize_history()}]
        structured_query = structure_query(query=question, context=context)
        
        query_cat = classify_query(structured_query)

        query_cat_lower = query_cat.lower()
        if query_cat_lower == "tavily":
            response = search_tavily(structured_query)
            messages = [
                {"role": "system", "content": f"""You are an AI Study Assistant. Answer the following question ONLY based on the notes provided to you. 
                If you can't find the answer in the notes, state that "I can't find the answer in the provided notes." Additional instructions: {style}."""},
                *[{"role": msg["role"], "content": msg["content"]} for msg in context],
                {"role": "user", "content": f"Notes: {response}\nQuestion: {structured_query}"}
            ]
            initial_answer = call_openai(messages)

        elif query_cat_lower == "wikipedia":
            response = search_wikipedia(structured_query)
            messages = [
                {"role": "system", "content": f"""You are an AI Study Assistant. Answer the following question ONLY based on the notes provided to you. 
                If you can't find the answer in the notes, state that "I can't find the answer in the provided notes." Additional instructions: {style}."""},
                *[{"role": msg["role"], "content": msg["content"]} for msg in context],
                {"role": "user", "content": f"Notes: {response}\nQuestion: {structured_query}"}
            ]
            initial_answer = call_openai(messages)

        elif query_cat_lower == "math":
            initial_answer = plan_execute_refine_math(structured_query, context)

        elif query_cat_lower == "rag":
            note_context = retrieve_chunks(structured_query, vectordb, k=5)
            messages = [
                {"role": "system", "content": f"""You are an AI Study Assistant. Answer the following question ONLY based on the notes provided to you. 
                If you can't find the answer in the notes, state that "I can't find the answer in the provided notes." Additional instructions: {style}."""},
                *[{"role": msg["role"], "content": msg["content"]} for msg in context],
                {"role": "user", "content": f"Notes: {note_context}\nQuestion: {structured_query}"}
            ]

            initial_answer = call_openai(messages)
            error_terms = ["error", "not found", "undefined", "can't find", "failed"]
            if any(term in initial_answer.lower() for term in error_terms):
                messages = build_domain_specific_prompt(structured_query, domain_embeds, context, style) 
            initial_answer = call_openai(messages)

        else:
            initial_answer = fallback_strategy(structured_query, context)

        confidence_scores = evaluate_confidence(structured_query, initial_answer, context, use_llm=True)

        if confidence_scores['final_confidence'] < 0.7:
            corrected_answer = self_correct_response(structured_query, initial_answer, context)
        else:
            corrected_answer = initial_answer
        
        add_to_history(role="Assistant", content=corrected_answer)
        is_complaint, final_answer = check_ethical_compliance(corrected_answer, query=False)

        print(final_answer)


if __name__ == "__main__":
    main()
