from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained('t5_final_model')
tokenizer = T5Tokenizer.from_pretrained('t5_final_tokenizer')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# This tiny script allows you to test our model - simply enter a context and run the script, the resulting output
# is the model's generated question based on this context; what you see is a sample context from the squad dataset

context = ("About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; however, all residence halls have 24-hour social spaces for students regardless of gender. Many residence halls have at least one nun and/or priest as a resident. There are no traditional social fraternities or sororities at the university, but a majority of students live in the same residence hall for all four years. Some intramural sports are based on residence hall teams, where the university offers the only non-military academy program of full-contact intramural American football. At the end of the intramural season, the championship game is played on the field in Notre Dame Stadium.")

input_text = f"generate question: {context}"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

output_ids = model.generate(input_ids, max_length=64)
prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(prediction)