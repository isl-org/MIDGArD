#ifndef Manifold_H_
#define Manifold_H_

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "Octree.h"
#include "BVH.h"
#include <map>
#include <queue>
#include <cstdlib>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/readSTL.h>

using namespace std;

// Manifold class for handling geometric mesh operations
class Manifold
{
public:
    // Nested structure for edge information
    struct Edge_Info
    {
        int face_x, face_y;
        std::map<int, int> loop_index;
    };
    vector<set<int>> v_faces;

    // Constructors and destructors
    Manifold();
    virtual ~Manifold();

    // Public interface
    int Load(char *filename); // Loads the model
    void Process_Manifold(int resolution);
    void SaveOBJ(const char *filename);          // Saves the mesh in OBJ format
    void Save(const char *filename, bool color); // Saves the mesh to a file

private:
    // Private interface
    void Calc_Bounding_Box();        // Calculates the bounding box of the mesh
    void Build_Tree(int resolution); // Builds an octree for the mesh
    void Construct_Manifold();
    void Project_Manifold();
    glm::dvec3 Closest_Point(const glm::dvec3 *triangle, const glm::dvec3 &sourcePosition);
    glm::dvec3 Find_Closest(int i);
    int is_manifold(); // Checks if the mesh is a manifold
    bool Split_Grid(map<Grid_Index, int> &vcolor, vector<glm::dvec3> &nvertices, vector<glm::ivec4> &nface_indices, vector<set<int>> &v_faces, vector<glm::ivec3> &triangles);

    // Utility methods
    inline double clamp(double d1, double l, double r)
    {
        return (d1 < l) ? l : ((d1 > r) ? r : d1);
    }

    // Member variables
    vector<Grid_Index> v_info;
    vector<glm::dvec3> vertices, vertices_buf;
    vector<glm::dvec3> colors;
    vector<glm::ivec3> face_indices, face_indices_buf;
    vector<glm::dvec3> face_normals;
    glm::dvec3 min_corner, max_corner;
    Octree *tree = nullptr;
    vector<BV *> bvs;
    char *fn;

    // Vector field
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXi N;
};

#endif
