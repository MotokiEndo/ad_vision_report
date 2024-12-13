from PIL import Image
from transformers import pipeline
import openai
import os

def fish_classification():
    pipe = pipeline("image-classification", model="jeemsterri/fish_classification")
    image = Image.open("image/fish.png")
    result = pipe(image)
    #print(result)

    # 最もスコアが高い結果を取得
    fish_result = max(result, key=lambda x: x['score'])

    # ラベルを出力
    print(f"Label with highest score: {fish_result['label']}")
    return fish_result['label']

def generate_recipe(fish):
    # Assign OpenAI API Key from environment variable
    openai.api_key = 'your api key'
    messages = []
    system_msg = "実在する魚の名前を与えます．この魚のレシピを考えてください"
    messages.append({"role": "system", "content": system_msg})

    messages.append ({"role": "user", "content": fish})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_token = 50
    )
    resipe = response["choices"][0]["message"]["content"]
    return resipe

def main():
    fish = fish_classification()
    recipe = generate_recipe(fish)
    print(recipe)

if __name__ == '__main__':
    main()