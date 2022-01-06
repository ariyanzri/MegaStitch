#include <iostream>
using namespace std;

#include "ceres/ceres.h"
using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#include "glog/logging.h"
#include <math.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <fstream>

using namespace std;


struct Reprojection_Single_Residual {
   template <typename T>

   bool operator()(T const* const* x, T* residual) const 
   {
      T a1 = x[0][img_1_index*9+0];
      T b1 = x[0][img_1_index*9+1];
      T c1 = x[0][img_1_index*9+2];
      T d1 = x[0][img_1_index*9+3];
      T e1 = x[0][img_1_index*9+4];
      T f1 = x[0][img_1_index*9+5];
      T g1 = x[0][img_1_index*9+6];
      T h1 = x[0][img_1_index*9+7];
      T k1 = x[0][img_1_index*9+8];

      T a2 = x[0][img_2_index*9+0];
      T b2 = x[0][img_2_index*9+1];
      T c2 = x[0][img_2_index*9+2];
      T d2 = x[0][img_2_index*9+3];
      T e2 = x[0][img_2_index*9+4];
      T f2 = x[0][img_2_index*9+5];
      T g2 = x[0][img_2_index*9+6];
      T h2 = x[0][img_2_index*9+7];
      T k2 = x[0][img_2_index*9+8];


      T det = a1 * (e1 * k1 - f1 * h1) - b1 * (d1 * k1 - f1 * g1) + c1 * (d1 * h1 - e1 * g1);

      if (det == T(0))
      {
        residual[0] = T(0);
        return true;
      }

      T A1 = (e1 * k1 - f1 * h1) / det;
      T B1 = -(b1 * k1 - c1 * h1) / det;
      T C1 = (b1 * f1 - c1 * e1) / det;
      T D1 = -(d1 * k1 - f1 * g1) / det;
      T E1 = (a1 * k1 - c1 * g1) / det;
      T F1 = -(a1 * f1 - c1 * d1) / det;
      T G1 = (d1 * h1 - e1 * g1) / det;
      T H1 = -(a1 * h1 - b1 * g1) / det;
      T K1 = (a1 * e1 - b1 * d1) / det;

      T A3 = A1*a2 + B1*d2 + C1*g2;
      T B3 = A1*b2 + B1*e2 + C1*h2;
      T C3 = A1*c2 + B1*f2 + C1*k2;
      T D3 = D1*a2 + E1*d2 + F1*g2;
      T E3 = D1*b2 + E1*e2 + F1*h2;
      T F3 = D1*c2 + E1*f2 + F1*k2;
      T G3 = G1*a2 + H1*d2 + K1*g2;
      T H3 = G1*b2 + H1*e2 + K1*h2;
      T K3 = G1*c2 + H1*f2 + K1*k2;

      T X = A3*T(P2_x) + B3*T(P2_y)+C3;
      T Y = D3*T(P2_x) + E3*T(P2_y)+F3;
      T W = G3*T(P2_x) + H3*T(P2_y)+K3;

     residual[0] = T(sqrt(pow(T(P1_x) - X/W,2) + pow(T(P1_y) - Y/W,2)));

     return true;
   }

   int img_1_index;
   int img_2_index;

   double P1_x,P1_y,P2_x,P2_y;

   Reprojection_Single_Residual(const int i,const int j,const double p1_x,const double p1_y,const double p2_x,const double p2_y)
   {
     img_1_index = i;
     img_2_index = j;
     P1_x = p1_x;
     P1_y = p1_y;
     P2_x = p2_x;
     P2_y = p2_y;
   }
};

double* get_homography_from_string(string text, int &index)
{
    double *h = (double*)malloc(sizeof(double)*9);

    char *cstr = new char[text.length() + 1];
    strcpy(cstr, text.c_str());

    char *token = strtok(cstr, " "); 
    
    int i = 0;

    index = stoi(token);

    token = strtok(NULL, " "); 

    while (token != NULL) 
    { 
        h[i] = stod(token);
        token = strtok(NULL, " "); 
        i++;
    } 

  return h;
}

double* get_matches_from_string(string text, double ****matches)
{
    char *cstr = new char[text.length() + 1];
    strcpy(cstr, text.c_str());

    char *token = strtok(cstr, " "); 
    int img_1_index = stoi(token);

    token = strtok(NULL, " "); 
    int img_2_index = stoi(token);

    token = strtok(NULL, " "); 
    int count_matches = stoi(token);

    for (int i = 0; i < count_matches; ++i)
    {
      token = strtok(NULL, " "); 
      double p1_x = stod(token);

      token = strtok(NULL, " "); 
      double p1_y = stod(token);

      token = strtok(NULL, " "); 
      double p2_x = stod(token);

      token = strtok(NULL, " "); 
      double p2_y = stod(token);

      matches[img_1_index][img_2_index][i][0] = p1_x;
      matches[img_1_index][img_2_index][i][1] = p1_y;
      matches[img_1_index][img_2_index][i][2] = p2_x;
      matches[img_1_index][img_2_index][i][3] = p2_y;

    }
    
}

void correct_absolute_homographies(string file_path,const char *arg)
{

  double **absolute_homographies;
  double ****matches;

  int num_images;

  fstream newfile;
  
  newfile.open(file_path,ios::in);
  if (newfile.is_open())
  {
     string line;
     getline(newfile, line);
     num_images = stoi(line);
     absolute_homographies = (double**)malloc(num_images*sizeof(double*));

     for (int i = 0; i < num_images; ++i)
     {
       getline(newfile, line);

       int index=-1;
       double *h = get_homography_from_string(line,index);

       absolute_homographies[index] = (double*)malloc(9*sizeof(double));
       absolute_homographies[index] = h;
     }

     getline(newfile, line);
     int num_matches = stoi(line);

     matches = (double ****)malloc(num_images * sizeof(double ***));

     for (int i = 0; i < num_images; ++i)
     {
        matches[i] = (double ***)malloc(num_images * sizeof(double **));

        for (int j = 0; j < num_images; ++j)
        {
          matches[i][j] = (double **)malloc(20 * sizeof(double*));   

          for (int k = 0; k < 20; ++k)
          {
            matches[i][j][k] = (double *)malloc(4 * sizeof(double));

            for (int m = 0; m < 4; ++m)
            {
                matches[i][j][k][m] = -1;
            }
          } 
        }
     }

     for (int i = 0; i < num_matches; ++i)
     {
       getline(newfile, line);

       get_matches_from_string(line,matches);
     }

     newfile.close();
  }


  google::InitGoogleLogging(arg);

  std::vector<double*> initial_x;
  double tmp[num_images*9];

  initial_x.push_back(tmp);

  for (int i = 0; i < num_images; ++i)
  {
    // printf("Image index : %d\n", i);
    // printf("\t Homography :");

    for (int j = 0; j < 9; ++j)
    {
      initial_x[0][i*9+j]=absolute_homographies[i][j];
      
      // printf("%f ", absolute_homographies[i][j]); 
    }
    // printf("\n");
  }

  Problem problem;


  for (int i = 0; i < num_images; ++i)
  {
    for (int j = 0; j < num_images; ++j)
    {
      for (int k = 0; k < 20; ++k)
      {

        if (matches[i][j][k][0] == -1 && matches[i][j][k][1] == -1 && matches[i][j][k][2] == -1 && matches[i][j][k][3] == -1)
        {
          continue;
        }

        double p1_x = matches[i][j][k][0];
        double p1_y = matches[i][j][k][1];
        double p2_x = matches[i][j][k][2];
        double p2_y = matches[i][j][k][3];

        DynamicAutoDiffCostFunction<Reprojection_Single_Residual, 4>* cost_function = 
          new DynamicAutoDiffCostFunction<Reprojection_Single_Residual, 4>(new Reprojection_Single_Residual(i,j,p1_x,p1_y,p2_x,p2_y));

        cost_function->AddParameterBlock(num_images*9);
        cost_function->SetNumResiduals(1);

        problem.AddResidualBlock(cost_function, nullptr, initial_x);

      }
    }
  }


  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";

  ofstream myfile (file_path);

  if (myfile.is_open())
  {
    myfile << num_images <<"\n";

    for (int i = 0; i <num_images; ++i)
    {
      myfile << i << " ";

      for (int j = 0; j < 9; ++j)
      {
        if (j<8)
        {
          myfile << initial_x[0][i*9+j] << " ";
        }
        else
        {
          myfile << initial_x[0][i*9+j];
        } 
      }

      myfile << "\n";
    }
    
    
    myfile.close();
  }

}


int main(int argc, char const *argv[])
{
    
  if (argc!=2)
  {
    printf("Not enough arguments. Path to the input file is needed.\n");
    return 0;
  }

  string path_input_file = argv[1]; 

  correct_absolute_homographies(path_input_file,argv[0]);

    return 0;
}

