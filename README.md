# learning_ai_with_art

> A tiny, and easy to use, implementation + collection of resources about AI&Art.

### Usage

Install [Docker](https://docs.docker.com/install/).

* Directly from [Dockerhub](https://hub.docker.com/r/p1nox/learning_ai_with_art):

```
docker run -it \
	-v INPUT_PATH:assets/inputs -v OUTPUT_PATH:assets/outputs \
	p1nox/learning_ai_with_art bash

# add input images in assets/inputs (as content and style),
# and output results will be created in assets/outputs
python -m neural_art -c assets/inputs/landscape.jpg -s assets/inputs/vgogh.jpg
```

* Repo as source:

```
git clone git@github.com:p1nox/learning_ai_with_art.git

cd learning_ai_with_art
sh build_image.sh
sh start_container.sh

# add input images in assets/inputs (as content and style),
# and output results will be created in assets/outputs
python -m neural_art -c assets/inputs/landscape.jpg -s assets/inputs/vgogh.jpg
```

### AI & Art sources and resources

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

* [Deep Learning & Art: Neural Style Transfer](https://sandipanweb.wordpress.com/2018/01/02/deep-learning-art-neural-style-transfer-an-implementation-with-tensorflow-in-python/)

* [Siraj collection on AI and Art](https://www.youtube.com/watch?v=9Mxw_ilpvwA&list=PL2-dafEMk2A5Y14yGVeBwDTbx0kt93Iae)

* [Painting like Van Gogh with Convolutional Neural Networks](http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks)

* [VGG19 - MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/)

* [Vincent: AI Artist](https://github.com/saikatbsk/Vincent-AI-Artist)

* [Deep Learning Course - Coursera](https://www.coursera.org/learn/neural-networks-deep-learning)

* [Creating art with deep neural networks](https://blog.paperspace.com/art-with-neural-networks/)

* [Computer algorithm can accurately identify Jackson Pollock paintings](https://arstechnica.com/science/2015/02/computer-algorithm-can-accurately-identify-jackson-pollock-paintings/)

* [When A Machine Learning Algorithm Studied Fine Art Paintings, It Saw Things Art Historians Had Never Noticed](https://medium.com/the-physics-arxiv-blog/when-a-machine-learning-algorithm-studied-fine-art-paintings-it-saw-things-art-historians-had-never-b8e4e7bf7d3e)

* [Is Art Created by AI Really Art?](https://www.scientificamerican.com/article/is-art-created-by-ai-really-art/)

* [Putting the art in artificial intelligence](http://www.dw.com/en/putting-the-art-in-artificial-intelligence/av-43008540)

* [Why AI is a revolution](https://www.youtube.com/watch?v=86X1Nln-PHw&feature=youtu.be)

* [JWT's 'The Next Rembrandt' Wins Two Grand Prix at Cannes, in Cyber and Creative Data](http://www.adweek.com/brand-marketing/jwts-next-rembrandt-wins-two-grand-prix-cannes-cyber-and-creative-data-172171/)

* [Delivering real-time AI in the palm of your hand](https://code.facebook.com/posts/196146247499076/delivering-real-time-ai-in-the-palm-of-your-hand/)

* [TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
