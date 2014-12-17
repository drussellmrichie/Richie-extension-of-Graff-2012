Richie-extension-of-Graff-2012
==============================

This repo contains code for a spreading activation model of word production that extends that of Graff (2012). Eventually, adjustments will be made to this model to (hopefully!) account for his finding that natural language lexicons tend to have more minimal pairs for more discriminable phonemic contrasts.

**Richie writeup of extension to Graff (2012).docx** describes my efforts so far, and gives a little more background to this effort.

**MinimalPairsAnalysis.py**, taking **confusability data.csv** and **english distinctive features.csv**, shows that more discriminable phonemic contrasts (discriminability data taken from Miller and Nicely (1955)) tend to differ on more distinctive features (features taken from [here](http://clas.mq.edu.au/speech/phonetics/phonology/features/)). It genereates **discriminability vs feature distance.png**. This finding forms the basis for the extension to the Graff model that might explain Graff's minimal pairs finding.

**SpeechProductionNetworkDiagram.py** can be used to display spreading activation networks of various architectures. It generates **network architecture.png**.

**SpeechProduction.py** simulates word production and lexicon evolution. This will generate a file average similarity over time.png, which plots the average similarity among word-forms at each step in simulation. This was a basic finding of Graff (2012)'s model (and his other empirical work). That the current model does not replicate this finding suggests certain modifications need to be made (suggested in the .docx file above). **SpeechProduction.py** will also generate .png's for the activation across the network at each time step. These can they be made to use gif's using the snippet at the end of **SpeechProduction.py**, which calls **images2gif.py**.
