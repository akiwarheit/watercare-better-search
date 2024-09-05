from search import find_most_relevant_page
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def pe(content, query, url):
    prompt = (
        "You are an assistant for Watercare, a company providing water services in Auckland. "
        "Only answer questions related to water supply, faults, and water-related services in Auckland.\n\n"
        "Example 1: Question: How do I report a leak? Answer: You can report a leak by calling Watercare at (09) 442 2222 or using the digital agent on our website.\n\n"
        "Example 2: Question: What are the current outages in my area? Answer: Check the Watercare outages page to see if there are any ongoing outages in your area.\n\n"
        "Example 3: Question: what to do if I have water discolouration? Answer: If you have discoloured water, turn on (flush) your outside tap for about 10-15 minutes to clear your private water pipes. Use the hose tap closest to the water meter. Don't use any water in the house while you do this."
        f"\n\n{content}\n\n"
        f"Question: {query}"
    )

    return prompt

def ask(question):
    page = find_most_relevant_page(question)

    if page != None:
        if page['score'] < 0.1: # This is arbitrary - we obviously need to run more tests
            return "Not relevant", None, None
        
        prompt = pe(page['content'], question, page['url'])

        input_ids = tokenizer(prompt, max_length=2048, truncation=True, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=2048)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response, page['url'], page['score']
    
    return "Not relevant", None, None