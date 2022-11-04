import mshr
from fenics import Point
from scipy.spatial import ConvexHull


# POLYGON MESHES
def polygon_geom(vertices, angle):
    """Creates mesh geometry in the shape of an arbitrary
    polygon defined by its vertices
    
    Arguments:
      vertices (2D-array): vertex coordinates in counter-clockwise order
      angle (float): angle in radians to rotate the z-axis
    """
    vertices = [Point(v) for v in vertices]
    geom = mshr.Polygon(vertices)
    geom = mshr.CSGRotation(geom, angle)
    return geom

def polygon_mesh(resolution, vertices, angle=0):
    """Creates mesh in the shape of an arbitrary
    polygon defined by its vertices
    
    Arguments:
      resolution (float): sets the mesh resolution
      vertices (2D-array): vertex coordinates in counter-clockwise order
      angle (float): angle in radians to rotate the z-axis
    """
    geom = polygon_geom(vertices, angle)
    return mshr.generate_mesh(geom, resolution)

def convex_hull_geom(points, angle):
    """Creates mesh geometry as a convex hull around given points
    
    Arguments:
      points (2D-array): points around the hull is created
    """
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    geom = polygon_geom(vertices, angle)
    return geom

def convex_hull_mesh(resolution, points, angle=0):
    """Creates mesh in the shape of a convex hull around given points
    
    Arguments:
      resolution (float): sets the mesh resolution
      points (2D-array): points around the hull is created
    """
    geom = convex_hull_geom(points, angle)
    return mshr.generate_mesh(geom, resolution)


# RECTANGLE MESHES
def rectangle_geom(width, height, center, angle):
    """Creates mesh geometry in the shape of a rectangle
    
    Arguments:
      width (float): rectangle width
      height (float): rectangle height
      angle (float): angle in radians to rotate the z-axis
    """
    bottom_left = Point(0,0)
    upper_right = Point(width, height)
    # rectangle center is at (width/2, height/2)
    geom = mshr.Rectangle(bottom_left, upper_right)
    geom = mshr.CSGTranslation(geom, Point(center) - Point(width/2, height/2))
    geom = mshr.CSGRotation(geom, angle)
    
    return geom

def rectangle_mesh(resolution, width, height, center=[0,0], angle=0):
    """Creates mesh in the shape of a rectangle
    
    Arguments:
      resolution (float): sets the mesh resolution
      width (float): rectangle width
      height (float): rectangle height
      angle (float): angle in radians to rotate the z-axis
    """
    geom = rectangle_geom(width, height, center, angle)
    return mshr.generate_mesh(geom, resolution)

def square_geom(side_length, center=[0,0], angle=0):
    return rectangle_geom(width=side_length, height=side_length, center=center, angle=angle)

def square_mesh(resolution, side_length, center=[0,0], angle=0):
    geom = square_geom(side_length, center, angle)
    return mshr.generate_mesh(geom, resolution)

def unit_square_geom(center=[0,0], angle=0):
    return square_geom(side_length=1, center=center, angle=angle)

def unit_square_mesh(resolution, center=[0,0], angle=0):
    geom = unit_square_geom(center, angle)
    return mshr.generate_mesh(geom, resolution)


# ELLIPSE MESHES
def ellipse_geom(x_semi, y_semi, center=[0,0], segments=17, angle=0):
    """Creates mesh geometry in the shape of an ellipse.

    Arguments:
      x_semi (float): the horizontal semi-axis
      y_semi (float): the vertical semi-axis
      center ([cx, cy]): the ellipse's center
      segments (int): approximate the ellipse with an n-gon where n=segments
      angle (float): angle in radians to rotate the z-axis
    """
    geom = mshr.Ellipse(Point(center), x_semi, y_semi, segments)
    geom = mshr.CSGRotation(geom, angle)
    return geom

def ellipse_mesh(resolution, x_semi, y_semi, center=[0,0], segments=17, angle=0):
    """Creates mesh in the shape of an ellipse.

    Arguments:
      resolution (float): sets the mesh resolution
      x_semi (float): the horizontal semi-axis
      y_semi (float): the vertical semi-axis
      center ([cx, cy]): the ellipse's center
      segments (int): approximate the ellipse with an n-gon where n=segments
      angle (float): angle in radians to rotate the z-axis
    """
    geom = ellipse_geom(x_semi, y_semi, center, segments, angle)
    return mshr.generate_mesh(geom, resolution)

def disc_geom(radius, center, segments=17):
    return ellipse_geom(radius, radius, center, segments, angle=0)

def disc_mesh(resolution, radius, center=[0,0], segments=17):
    geom = disc_geom(radius, center, segments)
    return mshr.generate_mesh(geom, resolution)

def unit_disc_geom(segments=17):
    return disc_geom(radius=1, center=[0,0], segments=segments)

def unit_disc_mesh(resolution, segments=17):
    geom = unit_disc_geom(segments)
    return mshr.generate_mesh(geom, resolution)


##### Local Mesh Refinement #####
# TODO

