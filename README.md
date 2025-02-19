# Postman Path Finder
 Postman Path Finder is an algorithm which finds the shortest path for Austrian postmans. It a demo code and is not implemented with real world data.
![Agent Route Animation](/animations/smallAnimation.gif)


## Functionality
 The input data is represented in a grid which each house having x,y coordinates. Streets connect the houses where each street has a weight how long it would take to drive over it. Also each house has a timeweight.
 The algotihm uses K-Means to cluster the houses based on how time consuming an area is. It then generates a path based on the shortest decisions. This path is then optimized to find the shortest path. A visualization is added to show the movements of the agents.


## Requirements

- Python 3.6 or higher
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [tqdm](https://tqdm.github.io/)