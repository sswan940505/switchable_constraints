#include <vector>
#include <fstream>
#include <Eigen/Core>
#include "error_term.h"

Sophus::SE3d ReadVertex(std::ifstream *fin) {
    double x, y, z, qx, qy, qz, qw;
    *fin >> x >> y >> z >> qx >> qy >> qz >> qw;
    Sophus::SE3d
            pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(x, y, z));
    return pose;
}

Sophus::SE3d ReadEdge(std::ifstream *fin) {
    double x, y, z, qx, qy, qz, qw;
    *fin >> x >> y >> z >> qx >> qy >> qz >> qw;
    Sophus::SE3d
            pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(x, y, z));

    double information;
    for (int i = 0; i < 21; i++)
        *fin >> information;
    return pose;
}

int main(int argc, char **argv) {
    typedef Eigen::aligned_allocator<Sophus::SE3d> sophus_allocator;
    std::vector<Sophus::SE3d, sophus_allocator> vertices;
    std::vector<std::pair<std::pair<int, int>, Sophus::SE3d>, sophus_allocator>
            edges_odom,edges_loop;
    std::vector<double> edges_swtich;

    std::ifstream fin("../test.g2o");
    std::string data_type;
    while (fin.good()) {
        fin >> data_type;
        if (data_type == "VERTEX_SE3:QUAT") {
            int id;
            fin >> id;
            vertices.emplace_back(ReadVertex(&fin));
        } else if (data_type == "EDGE_SE3:QUAT") {
            int i, j;
            fin >> i >> j;
            edges_odom.emplace_back(std::pair<int, int>(i, j), ReadEdge(&fin));
        }else if (data_type == "EDGE_SE3_SWITCHABLE") {
            int i, j, k;
            fin >> i >> j >> k;
            edges_loop.emplace_back(std::pair<int, int>(i, j), ReadEdge(&fin));
            edges_swtich.emplace_back(1);
        }
        fin >> std::ws;
    }

    std::vector<double> v_s;
    std::vector<double> v_gamma;

    ceres::Problem problem;

    for (auto e: edges_odom) {
        auto ij = e.first;
        auto i = ij.first;
        auto j = ij.second;
        auto &pose_i = vertices.at(i);
        auto &pose_j = vertices.at(j);

        auto edge = e.second;
        //odometry
        ceres::CostFunction *cost_function = OdometryFunctor::Creat(edge);
        problem.AddResidualBlock(cost_function,
                                 NULL, pose_i.data(), pose_j.data());
    }
    int idx = 0;
    for (auto e: edges_loop) {
        auto ij = e.first;
        auto i = ij.first;
        auto j = ij.second;
        auto &pose_i = vertices.at(i);
        auto &pose_j = vertices.at(j);

        auto edge = e.second;
        //loop closure
        ceres::CostFunction *cost_function = LoopClosureFunctor::Creat(edge);
        problem.AddResidualBlock(cost_function,
                                 NULL, pose_i.data(), pose_j.data(), &edges_swtich[idx]);

        ceres::CostFunction
                *cost_function1 = PriorFunctor::Create(1);
        problem.AddResidualBlock(cost_function1,NULL,&edges_swtich[idx]);

        //no robust
        //      problem.SetParameterBlockConstant(&v_s.back());

        problem.SetParameterLowerBound(&edges_swtich[idx],0,0);
        problem.SetParameterUpperBound(&edges_swtich[idx],0,1);
        idx++;
    }
    std::cout<<std::endl;
    for (auto &i: vertices) {
        std::cout<<"V "<<i.data()<<std::endl;
        problem.SetParameterization(i.data(), new LocalParameterizationSE3);
    }

    std::cout<<"VC "<<vertices.front().data()<<std::endl;
    problem.SetParameterBlockConstant(vertices.front().data());

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    //    for (auto i: vertices) {
    //        std::cout << i.matrix() << "\n" << std::endl;
    //    }

    int trues   = 0;
    int falses = 0;
    for (auto s: edges_swtich) {
        std::cout << s <<",";
        if(s>0.5) trues++;
        else falses++;
    }
    std::cout<<"TRUE : "<<trues<<", FALSE : "<<falses<<std::endl;

    return 0;
}
