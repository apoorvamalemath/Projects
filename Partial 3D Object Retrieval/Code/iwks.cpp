#include <algorithm>
#include <iostream>
#include <math.h>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>


// Laplace-Beltrami Operator
#include <pclbo/utils.h>
#include <pclbo/pclbo.h>


//principle curvature
#include <vector>
#include <pcl/point_types.h>
#include <pcl/features/principal_curvatures.h>


    
int main(int argc, char *argv[]) {

    FILE *fp= NULL;
    FILE *fp1 = NULL;

    fp1 = fopen("log.txt", "a");
    fp = fopen("iwks.txt", "w");

    int i=0;

    std::string filename = argv[1];
    std::cout << "Reading " << filename << std::endl;
    fprintf(fp1, "Reading %s\n", argv[1]); 

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    //Load point cloud data
    pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud);
    int pts =  cloud->points.size();
    std::cout << "Loaded " << cloud->points.size () << " points." << std::endl;
    fprintf(fp1, "Loaded %d\n", pts); 


    // Create the KdTree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdt(new pcl::search::KdTree<pcl::PointXYZ>());
    kdt->setInputCloud(cloud);

    //-------------------------------------------------------------------------
    // Compute the normals and concatenate them to the points
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(kdt);
    ne.setKSearch(10);

    ne.compute(*normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    //-------------------------------------------------------------------------
    // Compute the LBO
    pclbo::LBOEstimation<pcl::PointNormal, pcl::PointNormal>::Ptr
    lbo(new pclbo::LBOEstimation<pcl::PointNormal, pcl::PointNormal>());

    lbo->setInputCloud(cloud_with_normals);
    lbo->setCloudNormals(cloud_with_normals);
    lbo->compute(fp1);

    //-------------------------------------------------------------------------
    // Set the Principal curvature
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne2;
    ne2.setInputCloud(cloud);
    ne2.setSearchMethod(kdt);
    ne2.setKSearch(10);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals2 (new pcl::PointCloud<pcl::Normal>);
    ne2.compute (*cloud_with_normals2);

     // Setup the principal curvatures computation
     pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;

    // Provide the original point cloud (without normals)
     principal_curvatures_estimation.setInputCloud (cloud);

    // Provide the point cloud with normals
     principal_curvatures_estimation.setInputNormals (cloud_with_normals2);

    // Use the same KdTree from the normal estimation
     principal_curvatures_estimation.setSearchMethod (kdt);
     principal_curvatures_estimation.setRadiusSearch (1.0);

    // Actually compute the principal curvatures
     pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
     principal_curvatures_estimation.compute (*principal_curvatures);


    double* pc1;
    pc1 = new double[cloud->points.size()];


   for(i=0; i< principal_curvatures->points.size (); i++){

  	 pcl::PrincipalCurvatures descriptor = principal_curvatures->points[i];
 	 pc1[i] = descriptor.pc1;


   }

    //-------------------------------------------------------------------------
   

    // compute wks
    
      std::cout << "Computing energy scales...." << std::endl;
      fprintf(fp1, "Computing energy scales....\n");

      int N = 100;  
      int wks_variance = 5; 

      double alpha = 0.015;
      double absolute, maximum, b, a, d, step, sigma, large;
      int n_functions = 300;

      double E;


      double* cuberoot_E;
      cuberoot_E = new double[n_functions];

      double* e;
      e = new double[N];

     //Compute energy samples
      for(int i=0; i < n_functions; i++){
	E = lbo->eigenvalues(i);
	absolute= std::abs(E);
	maximum = std::max(absolute, 1e-6);
	cuberoot_E[i] = std::pow(maximum, 1/3.);
     }

      large=cuberoot_E[0];

	for(int i=0; i<n_functions; i++)
	{
		if(large<cuberoot_E[i])
		{
			large=cuberoot_E[i];
		}
	}
     
      
      b= large/1.02;
      a=cuberoot_E[1];
      step = (b-a) / (N-1);
      int j=0;
      while(a <= b) {
       e[j]=a;
       	a += step;
        j++;
     }
	 
     sigma=(e[1]-e[0])*wks_variance;
     int k;
     
      
      std::cout << "computing iwks...." << std::endl;
     fprintf(fp1, "Computing iwks....");

     for (size_t x = 0; x < pts; x++){
      for(k=0; k< N; k++){

          double sum = 0.0;
          double sum1=0.0;
          double iwks_feature;
          double Ce;


          for (int i = 0; i < n_functions; i++)
	 {
            Eigen::VectorXd psi = lbo->eigenfunctions.col(i);
            sum += exp(-pow((e[k]-cuberoot_E[i]), 2)/(2*pow(sigma, 2)));
            sum1 += pow(psi(x), 2)* exp((-pow((e[k]-cuberoot_E[i]), 2))/(2*pow(sigma, 2)));
         }

          Ce = 1/sum;
          iwks_feature = (Ce*sum1) + (pc1[x]*alpha);
          fprintf(fp, "%lf ", iwks_feature);


     }

	fprintf(fp, "\n");
    }

    delete[] e;
    delete[] cuberoot_E;
    delete[] pc1;
    std::cout << "done" << std::endl;
   fprintf(fp1, "done\n");

    std::cout << "--------------------------------------------------------------------------" <<std::endl;
    fprintf(fp1, "--------------------------------------------------------------------------\n");

    fclose(fp);
    return 0;

}
