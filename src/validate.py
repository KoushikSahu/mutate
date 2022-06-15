import evaluate
import torch
from config import Config


def validate(dl, model, verbose=False):
    references = list()
    predictions = list()
    bleu_metric = evaluate.load('bleu')

    with torch.no_grad():
        for batch in dl:
            text_batch = batch['text']
            paraphrase_batch = batch['paraphrase']
            batch_input_ids = batch['input_ids'].to('cuda')

            batch_output_ids = model.generate(batch_input_ids,
                                              num_beams=2,
                                              min_length=0,
                                              max_length=Config.max_length)

            for input_text, paraphrase_text, batch_output_id in zip(text_batch,
                                                                    paraphrase_batch,
                                                                    batch_output_ids):
                predicted_paraphrase = Config.tokenizer.decode(batch_output_id,
                                                               skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False)

                if verbose:
                    print(f'Text                 : {input_text}')
                    print(f'Predicted paraphrase : {predicted_paraphrase}')

                references.append(paraphrase_text)
                predictions.append(predicted_paraphrase)

    metrics = bleu_metric.compute(predictions=predictions,
                                  references=references)

    print(f'Bleu score: {metrics["bleu"]}')
