# jazz-arpeggiator
*ATIAM (Ircam) Machine Learning Project :*  
Implementing a Recurrent Variational Autoencoder trained on the RealBook Dataset.  
The goal is to generate a melody based on automatically generated jazz chord progressions

## Files you can run
VAE_MNIST.py: Training of a VAE on the MNIST dataset\n
VAE_Realbook.py: Training of a VAE on small chord sequences\n
RVAE_Realbook.py: Training of a RVAE on full pieces\n
transition_matrix.py : Computes the chord transition matrix and save it in .txt\n
confusion_matrix.ipynb: Plot the confusion matrix\n

## References
**Text-based LSTM networks for Automatic Music Composition**, Keunwoo Choi, George Fazekas, Mark Sandler, *1st Conference on Computer Simulation of Musical Creativity*, Huddersfield, UK, 2016 [arXiv](https://arxiv.org/abs/1604.05358#), [pdf](https://arxiv.org/pdf/1604.05358v1), [bib](https://scholar.googleusercontent.com/citations?view_op=export_citations&user=ZrqdSu4AAAAJ&s=ZrqdSu4AAAAJ:MXK_kJrjxJIC&citsig=AMstHGQAAAAAWIjj06BhKkBaBGcqMR__UBSLuabfKgOR&hl=en&cit_fmt=0)
