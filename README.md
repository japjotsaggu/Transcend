# Transcend
Essentially, a transformer is a deep learning model that comprises an encoder and a decoder, leveraging attention mechanisms to identify crucial elements within sequential inputs. Using '[Attention is all you need](https://arxiv.org/abs/1706.03762)' by Vaswani et al. as the foundation, Transcend constructs all compenents of a transformer model from scratch to build a neural machine translation model. The model is trained to act as a sentence based English to French translator. 

## Documentation 
To know more about the process, intricacies and the architecture, [Explore Documentation](https://docs.google.com/document/d/129vcTkC4QC5IEbgkc4V8UDmAVlj3XwShBoIUCMY_H6g/edit?usp=sharing)

## Attention is all we need 
The attention mechanism lies at the core of the transformer model. Using this mechanism, the transformer eliminates the need for using Recurrent Neural Networks (RNNs) that come with their own set of limitations. To know more about attention and the transformer model architecture, check out my blog post [here](https://medium.com/@japjotsaggu31/attention-architecture-breaking-down-the-transformer-model-c870640de47a).

![Picture 1](https://github.com/japjotsaggu/Transcend/assets/119132799/82e81659-34a8-4af6-9f74-36933538dd95)

## Architecture 

## Installation and usage 
1. Clone the repo
   
   ```git clone https://github.com/japjotsaggu/Transcend.git```

   This may take about a minute
   
3. 'cd' into directory
   
   ```cd Transcend```
   
4. Install the required dependencies
   
   ```pip install -r requirements.txt```
   
5. Run application
   
   ```python translator.py```

   - You will be prompted to enter a text in English.
   - The model generates translations in French 
