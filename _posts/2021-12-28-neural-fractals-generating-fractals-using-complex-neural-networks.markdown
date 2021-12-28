---
layout: post
title:  "Neural Fractals : Generating Fractals Using Complex Valued Neural Networks"
date:   2021-12-28
categories: mathematics deeplearning
---
Recurrent Neural Networks (RNN) have demonstrated a significant ability of learning complicated dynamics So I wanted to check can we use them to generate fractals?   
## Mandelbrot Set
A simple however not simplest way to see a fractal is trying to plot the members of the Mandelbrot set in the complex plane. But what is the Mandelbrot set? consider a complex function $f_c(z) = z^2 + c$ then a complex number c is a member of Mandelbrot set if the following sequence does not diverge to infinity.  
  
$$ f_c(0), f_c(f_c(0)), f_c(f_c(f_c(0))), ... $$  
  
if you try to draw the members in the complex plane you will see an interesting pattern with amazing self-similarities:  
  
![Mandelbrot Set](/assets/img/fractals/mandelbrot.jpg)  
  
It's cool. but what if we use more a sophisticated function $f_c$ ?
## A Complex Recurrent Neural Network
Instead of $f_c(z) = z^2 + c$ we can use a more complicated function. for example:  
  
$$g_c(z) = NN_{\theta}(z, c)$$
   
where $NN_{\theta}$ is a neural network parametrized by a set of parameters $\theta$. Now the sequence  

$$g_c(0), g_c(g_c(0)), g_c(g_c(g_c(0))), ...$$  
  
can be expressed using a recurrent neural network. Note since this is gonna be a complex function, the neural network should also be a complex-valued neural network (CVNN). Implementing a network with complex parameters is easy. See [this](https://www.linkedin.com/posts/amirabbas-asadi_pytorch-deeplearning-neuralnetworks-activity-6879646507478331392-Aeky) for an example. So I created a complex RNN and tried to plot the associated set. My method was very similar to what explained [here](https://yozh.org/2012/05/24/mset008/).
- First I created a complex neural network with random parameters $\theta$
- Then applying the network on a set of random numbers sampled uniformly from the complex plane repeatedly for about 50 iterations.
- I assigned each sample that remained in a boundary $|c| \le r$ to the corresponding pixel in the resulting image
.
  
I repeated the above procedure until collecting enough samples. After applying a few enhancements to the resulting image, the result was amazing!  
  
![Gray Level Fractal](/assets/img/fractals/bw.png)
  
After pseudo coloring and applying a few image enhancements the result was even more beautiful:  

![Fractal 2](/assets/img/fractals/2.png)

## Neural Fractals Gallery
So I decided to generated a few more neural fractals!

![Fractal 1](/assets/img/fractals/1.png)  

![Fractal 3](/assets/img/fractals/3.png)  

![Fractal 4](/assets/img/fractals/4.png)  

![Fractal 5](/assets/img/fractals/5.png)  

![Fractal 6](/assets/img/fractals/6.png)  

![Fractal 7](/assets/img/fractals/7.png)  

![Fractal 8](/assets/img/fractals/8.png)  

## References
This blog gave me great ideas for plotting the fractals. However my method differs a little :
- [https://yozh.org/2012/05/24/mset008/](https://yozh.org/2012/05/24/mset008/)  
    
To know more about complex-valued neural networks see the following  
  
- A Survey of Complex-Valued Neural Networks, Joshua Bassey, Lijun Qian, Xianfang Li