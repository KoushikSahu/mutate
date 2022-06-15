from tqdm import tqdm


def train(dl, model, optimizer, loss_fn):
    for batch in (pb := tqdm(dl)):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = output['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pb.set_description(f'Loss: {loss}')
