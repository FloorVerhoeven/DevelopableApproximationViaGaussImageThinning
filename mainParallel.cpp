#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/writeOFF.h>
#include <igl/per_vertex_normals.h>
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>
#include <igl/adjacency_list.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/barycenter.h>
#include <igl/massmatrix.h>
#include <igl/writeOBJ.h>
#include <fstream>
#include <cmath>
#include <array>
#include <omp.h>

#pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in)\
     initializer(omp_priv=Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))

void findRotations(const Eigen::MatrixXd& N0,
                   const Eigen::MatrixXd& N1,
                   std::vector<Eigen::Matrix3d>& rot) {
    
    const auto n = N0.rows();
    rot.resize(n);
    
    for(int i = 0; i < n; ++i) {
        Eigen::Vector3d n1 = N0.row(i);
        Eigen::Vector3d n2 = N1.row(i);
        Eigen::Vector3d v = n1.cross(n2);
        const double c = n1.dot(n2);
        
        if(c > -1 + 1e-8) {
            const double coeff = 1 / (1 + c);
            Eigen::Matrix3d v_x;
            v_x << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
            rot[i] = Eigen::Matrix3d::Identity() + v_x + coeff * v_x * v_x;
        } else{
            rot[i] = -Eigen::Matrix3d::Identity();
        }
    }
}


std::vector<std::vector<int>> collectNeighbours(const std::vector<std::vector<int>>& adj,
                                                const Eigen::MatrixXd& V,
                                                const Eigen::MatrixXd& N,
                                                const double &rSquared,
                                                const double &nr,
                                                const std::vector<std::vector<int>> &prev_nbhs) {
    std::vector<std::vector<int>> results(V.rows());
    const double normalConeThreshold = cos(nr * M_PI / 180.);
    int max_thread =  omp_get_max_threads();

    std::vector<int> flags[max_thread];
    std::vector<int> stacks;
    #pragma omp parallel for
    for (int i=0; i<max_thread; i++){
        flags[i].resize(V.rows(), -1);
    }
    #pragma omp parallel for private(stacks)
    for(int i = 0; i < V.rows(); ++i) {
        const int t_id = omp_get_thread_num();
        stacks.clear();
        std::vector<int> result;
        result.reserve(prev_nbhs[i].size());
        stacks.push_back(i);
        flags[t_id][i] = i;

        while(!stacks.empty()) {
            auto id = stacks.back();
            stacks.pop_back();
            result.push_back(id);
            for (int j : adj[id]) {
                if(flags[t_id][j] != i && (N.row(i).dot(N.row(j))) > normalConeThreshold && (V.row(i) - V.row(j)).squaredNorm() < rSquared ) {
                    stacks.push_back(j);
                    flags[t_id][j] = i;
                }
            }
        }
        results[i].swap(result);
    }
    return results;
   
}

void fitNormals(const std::vector<std::vector<int>>& nbh,
                const Eigen::MatrixXd& V,
                const Eigen::MatrixXd& N,
                Eigen::MatrixXd& N2,
                const double cosineThreshold,
                const double sigma = 1.) {
    
    const auto nv = nbh.size();
    N2.resize(nv, 3);
    double angleThreshold = cosineThreshold * M_PI / 180.;

    #pragma omp parallel for
    for(int i = 0; i < nv; ++i) {
        const auto& nbi = nbh[i];
        Eigen::MatrixXd NN(nbi.size(), 3);

        for (int k = 0; k < nbi.size(); ++k) {
            NN.row(k) = N.row(nbi[k]);
        }

        Eigen::DiagonalMatrix<double, -1> W(nbi.size());
        if(sigma < 10.) {
            for(int i = 0; i < W.diagonal().size(); ++i) {
                double dot = NN.row(0).dot(NN.row(i));
                if (dot >= 1.){
                    W.diagonal()(i) = 1;
                } else if(dot < 0) {
                    W.diagonal()(i) = 0;
                } else {
                    W.diagonal()(i) = std::exp(-std::pow(acos(dot) / angleThreshold / sigma, 2));
                }
            }
        } else {
            W.diagonal().setOnes();
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(NN.transpose() * W * NN, Eigen::ComputeFullV);
        Eigen::Matrix3d frame = svd.matrixV();
        N2.row(i) = (frame.leftCols(2) * frame.leftCols(2).transpose() * N.row(i).transpose()).normalized();
    }
    
}

void assembleRHS(const Eigen::MatrixXd& C,
                 const Eigen::MatrixXd& V,
                 const Eigen::MatrixXi& F,
                 const std::vector<Eigen::Matrix3d>& R,
                 Eigen::MatrixXd& rhs) {
    
    const auto nv = V.rows();
    rhs.resize(nv, 3);
    rhs.setZero();

    #pragma omp parallel for reduction(+:rhs)
    for(int i = 0; i < F.rows(); ++i)  {
        for(int j = 0; j < 3; ++j)  {
            int v0 = F(i, (j + 1) % 3);
            int v1 = F(i, (j + 2) % 3);
            Eigen::Vector3d b = C(i,j) * R[i] * (V.row(v0) - V.row(v1)).transpose();
            rhs.row(v0) -= b.transpose();
            rhs.row(v1) += b.transpose();
        }
    }
   
}

std::vector<std::vector<int>> triangleAdjacency(const Eigen::MatrixXi& F, const size_t nv) {
    
    std::vector<std::vector<int>> vnbhs(nv);
    const auto nf = F.rows();
    
    for(int i = 0; i < nf; ++i) {
        for(int j = 0; j < 3; ++j) {
            vnbhs[F(i, j)].push_back(i);
        }
    }
    std::vector<int> flags(nf, -1);
    std::vector<std::vector<int>> ret(nf);
    
    #pragma omp parallel for
    for(int i = 0; i < nf; ++i) {
        for(int j = 0; j < 3; ++j) {
            for(int k : vnbhs[F(i, j)]) {
                if(k != i && flags[k] != i) {
                    ret[i].push_back(k);
                    flags[k] = i;
                }
            }
        }
    }
    return ret;
}

void center(Eigen::MatrixXd& V) {
    V.rowwise() -= V.colwise().mean();;
    V /= 2. * V.rowwise().norm().maxCoeff();
}

void gaussThinning(const std::string &mesh_folder,
                   const Eigen::MatrixXd &V_in,
                   const Eigen::MatrixXi &F,
                   Eigen::MatrixXd &V,
                   const int number_iterations = 100,
                   double minConeAngle = 2.5,
                   double smooth = 1e-5,
                   double start_angle = 25,
                   double radius = 0.1,
                   double sigma = 2.) {
    
    double coneAngle = start_angle;
    double r = radius;
    double r_squared = r*r;
    double eps = 1e-3;
    
    V = V_in;
    const auto nv = V.rows();
    center(V);
    
    igl::writeOFF(mesh_folder + "/normalized.off", V, F);
    
    Eigen::SparseMatrix<double> I(nv, nv);
    I.setIdentity();
    Eigen::MatrixXd B, b, C, N, N2;
    std::vector<Eigen::Matrix3d> rot;
    Eigen::SparseMatrix<double> L, M;
    std::vector<std::vector<int>> nbhs(F.rows(), std::vector<int>(1));
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol;
    auto tt = triangleAdjacency(F, nv);
    igl::cotmatrix_entries(V, F, C);
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
    
    if(smooth) {
        chol.compute(-L + smooth * L.transpose() * L + eps * M);
    } else {
        chol.compute(-L + eps * M);
    }
    
    for(int k = 0; k < number_iterations; ++k) {
        igl::per_face_normals(V, F, N);
        igl::barycenter(V, F, B);
        nbhs = collectNeighbours(tt, B, N, r_squared, coneAngle, nbhs);
        if(coneAngle > minConeAngle) coneAngle *= .95;
        fitNormals(nbhs, V, N, N2, coneAngle, sigma);
        findRotations(N, N2, rot);
        assembleRHS(C, V, F, rot, b);
        V = chol.solve(eps * M * V - b);
    }
    return;
}

void runExperiment(std::string folder, std::string inputFile, std::string outputFile, const int iters, const double minAngle, const double start_angle = 25, const double radius = 0.1, const double smooth = 1e-5, const double sigma = 2) {
    Eigen::MatrixXd V_in, V_out;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh(folder + "/" + inputFile, V_in, F);
    gaussThinning(folder, V_in, F, V_out, iters, minAngle, smooth, start_angle, radius, sigma);
    std::cout << folder << ": done" << std::endl;
    igl::write_triangle_mesh(folder + "/" + outputFile, V_out, F);
}

int main(int argc, const char * argv[]) {
    Eigen::initParallel();
    
    if(argc < 6) {
        std::cout << "Need input file, output file, output directory, number of iterations and minimum search cone. Running default experiments..." << std::endl;
       
        /* run default experiments here .... */
        runExperiment("./examples/architecture", "input.off", "out.obj", 100, 2.5);

        runExperiment("./examples/boat", "input.off", "out.obj", 100, 2.5);

        runExperiment("./examples/bumpy", "input.off", "out.obj", 150, 7.5);
        
        runExperiment("./examples/bunny", "input.off", "out.obj", 500, 2.5);

        runExperiment("./examples/bunny_high", "input.off", "out.obj", 100, 2.5);

        runExperiment("./examples/bunny_small", "input.off", "out.obj", 500, 5.0);

        runExperiment("./examples/coffee", "input.off", "out.obj", 500, 2.5);

        runExperiment("./examples/cone", "input.off", "out.obj", 100, 2.5);

        runExperiment("./examples/cone_high", "input.off", "out.obj", 100, 2.5, 25, 0.015);

        runExperiment("./examples/curved_fold", "input.off", "out.obj", 100, 2.5);

        runExperiment("./examples/cylinder", "input.off", "out.obj", 300, 7.5);

        runExperiment("./examples/dog", "input.off", "out.obj", 100, 5.0);

        runExperiment("./examples/dome", "input.off", "out.obj", 100, 7.5);

        runExperiment("./examples/dress_high", "input.off", "out.obj", 100, 7.5);

        runExperiment("./examples/drill", "input.off", "out.obj", 100, 7.5);

        runExperiment("./examples/einstein", "input.off", "out.obj", 300, 7.5, 60, 0.015);

        runExperiment("./examples/face", "input.off", "out.obj", 100, 5.0);

        runExperiment("./examples/fandisk", "input.off", "out.obj", 1000, 5.0);

        runExperiment("./examples/fertility", "input.off", "out.obj", 100, 7.5);

        runExperiment("./examples/guitar", "input.off", "out.obj", 100, 2.5);

        runExperiment("./examples/lilium", "input.off", "out.obj", 100, 5.0);

        runExperiment("./examples/mask", "input.off", "out.obj", 500, 2.5);

        runExperiment("./examples/nut", "input.off", "out.obj", 100, 2.5);

        runExperiment("./examples/swing", "input.off", "out.obj", 500, 5.0);

        runExperiment("./examples/washington", "input.off", "out.obj", 500, 5.0, 30, 0.045);
    } else
    {
        std::string  infile = argv[1];
        std::string  outfile = argv[2];
        std::string  folder = argv[3];
        
        auto numIters = std::atoi(argv[4]);
        auto minAngle = std::stold(argv[5]);
        auto start_angle = 25;

        std::cout << "Running on " << omp_get_max_threads() << " threads." <<std::endl;
        std::cout << "Processing " << infile << " with " << numIters << " iterations and mimimum cone angle " << minAngle << ". Output directory is " << folder << std::endl;
        runExperiment(folder, infile, outfile, numIters, minAngle, start_angle);
    }
    
    return 0;
}
