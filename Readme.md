
# Quick Example
**Example Construction:**

> **Question API; [https://hercai.onrender.com/v3/hercai?question=](https://hercai.onrender.com/v3/hercai?question=)**

**Example Question For Python:**

# Available Models 
# "v3", "v3-32k", "turbo", "turbo-16k", "gemini", "llama3-70b", "llama3-8b", "mixtral-8x7b", "gemma-7b", "gemma2-9b"
# Default Model: "v3"
# Premium Parameter: personality => Optional
question_result = herc.question(model="v3", content="hi, how are you?")
print(question_result)
# print(question_result["reply"]) For Reply
```

> **Text To Image API; [https://hercai.onrender.com/v3/text2image?prompt=](https://hercai.onrender.com/v3/text2image?prompt=)**

**Example Draw Image For Python:**

# Available Models 
# "v1", "v2", "v2-beta", "v3" (DALL-E), "lexica", "prodia", "simurg", "animefy", "raava", "shonin"
# Default Model: "v3"
image_result = herc.draw_image(model="simurg", prompt="A beautiful landscape", negative_prompt="Dark and gloomy")
print(image_result)
# print(image_result["url"]) For Image URL        

question_result = herc.beta_question(...)

image_result = herc.beta_draw_image(...)
