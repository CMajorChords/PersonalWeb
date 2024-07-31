
<h2 style='pointer-events: none;'>Basic Matrix Theory</h2>
<h3 style='pointer-events: none;'>1.What are vectors and matrices</h3>

Imagine an n-dimensional space where a vector is actually an arrow starting from the origin in n-dimensional space,
And the matrix actually represents a transformation, such as rotation, scaling, translation, and so on,
A vector is actually a linear combination of basis vectors on n dimensions,
So the question arises, what is the basis vector? The basis vector is a unit vector, for example, in two-dimensional space, the basis vectors are (1,0) and (0,1),
So a vector $(2,3) $is actually $2 * (1,0)+3 * (0,1) $,
Actually, 2 and 3 here are the projections of vectors on the basis vector,
It can also be said that it is the coordinates of the vector on the basis vector,
But is the base vector necessarily $(1,0) $and $(0,1) $, not necessarily,
The basis vector can be transformed, for example, in two-dimensional space, we can transform $(1,0) $and $(0,1) $into $(1,1) $and $(1,1, -1) $,
So $(2,3) $can be expressed as $2 * (1,1)+1 * (1, -1) $, which is $(3,1) $,
This transformation is the rotation, scaling, translation, and so on of the vector $(2,3) $,
By transforming the basis vector, we can perform various transformations on the vector, because the vector is actually a linear combination of the basis vectors,
So how can this transformation be represented by numbers? This is the function of matrices,
A matrix is used to represent how to operate a basis vector,
For example, we transform $(1,0) $and $(0,1) $into $(1,1) $and $(1,1, -1) $,
So we can use a matrix to represent this transformation, which is:

|1 | 1|
|---|---|
|1 | -1|

The first column of this matrix is $(1,1) $, and the second column is $(1, -1) $,
So the function of this matrix is to transform $(1,0) $into $(1,1) $, and $(0,1) $into $(1, -1) $,
Multiplying this matrix by a vector to the left is the transformation of this vector,
<h3 style='pointer-events: none;'>2. What is the inverse of a matrix</h3>

What is the inverse of a matrix? The inverse of a matrix is a matrix whose function is to invert the transformation of a matrix,
For example, the function of a matrix is to rotate a vector in a two-dimensional space by 90 degrees,
So the inverse of this matrix is to rotate a vector in a two-dimensional space by -90 degrees,
If we know the result of multiplying a matrix by a vector to the left, then we can find the vector by the inverse of the matrix,
That is, by performing the inverse operation of matrix operations on the results, the original vector is obtained,
For example, if the result of multiplying a matrix left by a vector is $(1,1) $, then we can find the original vector by $(1,1) $and the inverse of the matrix,
However, note that if the operation represented by a matrix is to compress the space into a low dimensional space, then the matrix has no inverse matrix,
Because this operation resulted in dimensional loss, all information on the lost dimension was lost,
So we cannot find the original vector through the inverse of this matrix,
Because you cannot infer the value of the original vector in the missing dimension,
<h3 style='pointer-events: none;'>3. What is the determinant</h3>

A determinant is a number that represents the impact of a matrix transformation on space,
For example, if the determinant of a matrix is 2, then the transformation of this matrix will magnify the area (volume) of space by two times,
For example, in a two-dimensional space, if the determinant of a matrix is 2, then the transformation of this matrix will magnify the area of the space by two times,
For example, in a three-dimensional space, if the determinant of a matrix is 2, then the transformation of this matrix will double the volume of the space,
A matrix with a determinant of 0 will compress the space into a low dimensional space,
For example, in a two-dimensional space, if the determinant of a matrix is 0, then the transformation of this matrix will compress the space onto a straight line,
At this point, the transformation of this matrix is no longer an invertible transformation,
<h3 style='pointer-events: none;'>4. What is the rank of column space and matrix</h3>

A column space is a vector space, and the basis vector of this vector space is the column vector of a matrix,
For example, if the column vectors of a matrix are $(1,0) $and $(0,1) $, then the column space of the matrix is a two-dimensional space,
Because the column vectors of this matrix are the basis vectors of a two-dimensional space,
But if the column vectors of a matrix are $(1,1) $and $(1, -1) $, then the column space of the matrix is a straight line,
Because this matrix compresses two-dimensional space onto a straight line,
The rank of a matrix is the dimension of its column space,
For example, if the column vectors of a matrix are $(1,0) $and $(0,1) $, then the rank of the matrix is 2,
Because the column space of this matrix is a two-dimensional space,
But if the column vectors of a matrix are $(1,1) $and $(1, -1) $, then the rank of the matrix is 1,
Because the column space of this matrix is a straight line and a one-dimensional space,
So what is the relationship between the rank and determinant of a matrix? If the determinant of a matrix is 0, it means that the matrix compresses the space into a low dimensional space,
So the rank of this matrix is the dimension of this low dimensional space,
So what are the dimensions of 0 space? The dimensions of 0 space are 0,
All column spaces contain 0 spaces, which is obvious,
<h3 style='pointer-events: none;'>5. Non square matrix</h3>

As mentioned earlier, each column of a matrix represents a transformation of the basis vector of a dimension,
So if the column vectors of a matrix are $(2,1) $and $(1,2) $, then the function of this matrix is to transform the vectors in two-dimensional space into $(2,1) $and $(1,2) $,
But if a matrix is

|1 | 1 | 1|
|---|---|---|
|1 | 2 | 3|

So the column vectors of this matrix are $(1,1) $and $(1,1) $and $(1,1) $. So what is the purpose of this matrix,
The function of this matrix is to transform the basis vectors of the three dimensions in three-dimensional space into $(1,1) $, $(1,1) $, and $(1,1) $, respectively,
So you may find that each basis vector only occupies the first and second dimensions, while the third dimension does not,
That is to say, the function of this matrix is to compress the three-dimensional space into the two-dimensional space,
So the determinant is naturally 0,
The rank of a matrix is 2,
Different from the following matrix:

|1 | 1 | 1|
|---|---|---|
|1 | 2 | 3|
|0 | 0 | 0|

The column vectors of this matrix are $(1,1,0) $and $(1,1,0) $and $(1,1,0) $,
In the third dimension, the component is 0, which means there are three dimensions in space,
However, the column space of the matrix is two-dimensional and has no components in the third dimension, but the third dimension exists,
<h3 style='pointer-events: none;'>6. Point product and duality</h3>

Dot product is the multiplication and addition of the corresponding elements of two vectors. Its geometric meaning is to project one vector onto another vector, and then multiply the length of the projection by the length of the other vector,
For example, the dot product of vector $(1,1) $and vector $(1,0) $is 1, and the dot product of vector $(1,1) $and vector $(0,1) $is 1,
If you find this operation difficult to understand, then we can understand vector dot products in a different way,
There is a unit vector $(√ 2/2, √ 2/2) $. If we project any vector onto $(√ 2/2, √ 2/2) $, then in fact, we first project the two base vectors onto $(√ 2/2, √ 2/2) $,
Multiplying the linear combination represented by vectors by these two projected basis vectors can be represented by a matrix:

|√ 2/2 | √ 2/2|
|---- | ----|

Did you find that this matrix is the transposition of the original unit vector,
If it is a non unit vector, then this non unit vector can be seen as a scaling of the original unit vector,
If this new vector is $(√ 2, √ 2) $, then this vector is scaled twice as much as the original unit vector,
The dot product with this vector can be seen as first projecting onto the unit vector and then scaling twice,
The corresponding matrix is:

|√ 2 | √ 2|
|---- | ----|

This matrix is the transposition of the original unit vector and multiplied by 2,
That's why dot product can be interpreted as first projecting in one direction, and then multiplying the value of the projection by the length of that direction,
The inspiration here is that as long as you see a linear transformation from n-dimensional to 1-dimensional, then this transformation can be represented by a matrix of n rows and 1 column,
This transformation corresponds to a vector.
This is a kind of duality that gives people goosebumps,
In other words, the duality of a vector can be seen as a linear transformation,
The duality of a linear transformation from a multidimensional space to a one-dimensional space is a specific vector of that one-dimensional space,
This indicates the essence of a vector: a vector is not just an arrow, it can also be seen as a material carrier of linear transformation.
Its meaning is that imagining a vector is easier than imagining the entire space moving on the number axis.