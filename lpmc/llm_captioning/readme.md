# Tag-to-Caption Generation using Large Language Model

This project aims to generate captions for music using existing tags. We leverage the power of OpenAI's GPT-3.5 Turbo API to generate high-quality and contextually relevant captions based on music tags.

## Installation
To run this project locally, follow the steps below:

```
pip install -e
# our exp version (date): openai-0.27.8, python-dotenv-1.0.0  (2023.04 ~ 2023.05)
```

Set up your OpenAI API credentials by creating a `.env` file in the root directory. Check [OpenAI API](https://openai.com/blog/openai-api) access and price first. Add the following lines and replace YOUR_API_KEY with your actual API key:

```bash
OPENAI_API_KEY=your_key_in_here
api_host=0.0.0.0
api_port=8088
```

## Caption Generation from Existing Tag Dataset
We have prepared randomly sampled 20 `tag-caption` pairs from [Music Caps](https://huggingface.co/datasets/google/MusicCaps). If you want to reproduce the 2.2 Miliion examples from the paper, download annotation data & track split from [MSD-ECALS](https://zenodo.org/record/7107130), [Magnatagatune-MTT](https://github.com/seungheondoh/msu-benchmark). 

```bash
python generate.py --dataset_type {musiccaps} --prompt {writing, summary, paraphrase, attribute_prediction}
```

tag_to_caption generation `writing` results:
```
query: 
write a song description sentence including the following attributes
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"Experience the raw and authentic energy of an amateur recording as mesmerizing beatbox rhythms intertwine with catchy finger snipping, while a soulful male voice delivers heartfelt lyrics on a medium tempo track."
```

tag_to_caption generation `summary` results:
```
query: 
write a single sentence that summarize a song with the following attributes. Don't write artist name or album name.
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"An amateur recording featuring a male voice singing over a medium tempo beatbox with finger snapping."
```

tag_to_caption generation `paraphrase` results:
```
query: 
write a song description sentence including the following attributes. paraphraze paraphrasing is acceptable.
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"This song features the mesmerizing combination of beatbox and finger snipping, accompanied by a captivating male voice. Recorded in an amateur setting, this medium-tempo track exudes unique charm and raw emotion."
```

tag_to_caption generation `attribute_prediction` results:
```
query: 
write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following attributes. for description, write a song description sentence including the following attributes and new attributes."
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
{
  "new_attribute": [
    "acoustic guitar",
    "uplifting lyrics",
    "catchy melody"
  ],
  "description": "A vibrant song featuring beatbox, finger snipping, a male voice, and an amateur recording. With its medium tempo, this track also incorporates elements of acoustic guitar, which add a melodic and uplifting touch. The catchy melody and heartfelt lyrics make it an enjoyable listening experience for anyone looking for a feel-good tune."
}
```


## Reproduce Tag-to-Caption Generation Results using huggingface dataset


```bash
pyhon eval.py --python eval.py --prediction_col {caption_writing, caption_summary, caption_paraphrase, caption_attribute_prediction}
```
`caption_writing` results

```yaml
{
    "bleu1": 0.3683557393417835,
    "bleu2": 0.19845761130017955,
    "bleu3": 0.1137270541209299,
    "bleu4": 0.06735589118343131,
    "meteor_1.0": 0.31436851912375435,
    "rougeL": 0.25362155449947565,
    "bertscore": 0.892640920099384,
    "vocab_size": 5521,
    "vocab_diversity": 0.5616736098532874,
    "caption_novelty": 1.0
}
```
