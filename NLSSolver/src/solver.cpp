
/*
*求解 y=exp(ax^2+bx+c)
*/

#include <iostream>
#include <random>
#include <memory>

#include <eigen3/Eigen/Dense>

#include "tic_toc.h"

using namespace std;

//损失函数基类
class LossFuntion
{
public:
    LossFuntion(double weight)
        : weight_(weight)
    {}
    virtual void computeJacobian(double u_x) = 0;

    double weight_;
    double jacobian_;
};

//柯西损失函数
class CauchyLoss : public LossFuntion
{
public:
    CauchyLoss(double weight)
        : LossFuntion(weight)
    {}

    void computeJacobian(double u_x)  override
    {
        jacobian_ = weight_* u_x / (1.0 + u_x * u_x);
    }
};

//顶点类
class Vertex
{
public:
    Vertex(int dim)
        : dimension_(dim)
    {
        vertex_id_ = static_id ++;
        ordering_id_ = 0;
        data_.resize(dimension_);
    }
    
    Eigen::VectorXd data_;      //该顶点存储的数据内容
    unsigned long vertex_id_;          //该顶点的id
    int dimension_;             //该顶点的维度
    unsigned long ordering_id_;
    static unsigned long static_id;
};

unsigned long Vertex::static_id = 0;

//约束边基类
class Factor
{
public:
    Factor() 
        : loss_function_(NULL)
    {}

    void setParametersDims(int dim_residual,vector<std::shared_ptr<Vertex>>& vs)
    {
        dim_residual_ = dim_residual;
        vertices_ = vs;
        error_.resize(dim_residual_);
        jacobians_.resize(vs.size());
    }

    virtual void computeError() = 0;
    
    virtual void computeJacobian() = 0;

    void setInformationMatrix(Eigen::MatrixXd information)
    {
        information_ = information;
    }

    void setLossFunction(std::shared_ptr<LossFuntion> loss)
    {
        loss_function_ = loss;
    }

    vector<std::shared_ptr<Vertex>> vertices_;
    Eigen::VectorXd error_;
    double squared_error_;
    vector<Eigen::MatrixXd> jacobians_;
    int dim_residual_;
    Eigen::VectorXd  meas_;
    Eigen::MatrixXd information_;
    Eigen::MatrixXd robust_information_;
    std::shared_ptr<LossFuntion> loss_function_;
};

//曲线拟合约束边
class CurveFittingFactor : public Factor
{
public:
    CurveFittingFactor(Eigen::VectorXd meas)
    {
        meas_ = meas;
    }
    void computeError()  override
    {
        
        Eigen::MatrixXd new_information = information_;
        double a_op = vertices_[0]->data_(0);
        double b_op = vertices_[0]->data_(1);
        double c_op = vertices_[0]->data_(2);
        error_(0) = std::exp(a_op*meas_(0)*meas_(0)+b_op*meas_(0)+c_op) - meas_(1);
        if(loss_function_)
        {
            double u_x = sqrt(error_.transpose()*information_*error_);
            loss_function_->computeJacobian(u_x);
            robust_information_ = 1.0 / (u_x+0.000001) * loss_function_->jacobian_ * information_;
            new_information = robust_information_;
        }
        squared_error_ = 0.5 * error_.transpose()*new_information*error_;
    }
    
    void computeJacobian()  override
    {
        Eigen::MatrixXd jacobian_abc(dim_residual_,vertices_[0]->dimension_);
        double a_op = vertices_[0]->data_(0);
        double b_op = vertices_[0]->data_(1);
        double c_op = vertices_[0]->data_(2);
        double exp_x = std::exp(a_op*meas_(0)*meas_(0)+b_op*meas_(0)+c_op);
        jacobian_abc << meas_(0)*meas_(0)*exp_x, meas_(0)*exp_x, exp_x;
        jacobians_[0] = jacobian_abc;
    }

};

class LSSolver
{
public:
    LSSolver()
        : currentChi_(0.0)
    {}

    bool solve(int iterations)
    {
        if(factors_.size() == 0 || vertices_.size() == 0)
        {
            std::cerr << "\n cannot solve problem without edges or vertices" << std::endl;
            return false;
        }
        TicToc t_solve;
        setOrdering();
        makeHessian();
        int iter = 0;
        while(iter < iterations)
        {
            //计算当前总的残差
            currentChi_ = 0.0;
            for(auto factor : factors_)
            {
                currentChi_ += factor->squared_error_;
            }
            std::cout << "iter: " << iter << ", " << "currentChi: " << currentChi_ << std::endl;
            solveLinearSystem();
            if(delta_x_.squaredNorm() <= 1e-6 )
                break;
            updateStates();     //update state 
            makeHessian();
            iter++;
        }
        
        std::cout << "problem solve cost: " << t_solve.toc() << "ms" << std::endl;
        return true;
    }

    void solveLinearSystem()
    {
//         delta_x_ = Hessian_.inverse() * b_;
        delta_x_ = Hessian_.ldlt().solve(b_);
    }

    void updateStates()
    {
        for(auto &vertex : vertices_)
        {
            ulong idx = vertex->ordering_id_;
            ulong dim = vertex->dimension_;
            Eigen::VectorXd delta = delta_x_.segment(idx,dim);
            vertex->data_ += delta;
        }
    }

    void makeHessian()
    {
    
        Eigen::MatrixXd H(all_states_size,all_states_size);
        Eigen::VectorXd b(all_states_size);
        H.setZero();
        b.setZero();

        for(auto& factor: factors_)
        {
            factor->computeError();
            factor->computeJacobian();
            auto jacobians = factor->jacobians_;
            auto verticies = factor->vertices_;
            assert(jacobians.size() == verticies.size());
            for(size_t i = 0; i < verticies.size(); ++i)
            {
                auto v_i = verticies[i];
                auto jacobia_i = jacobians[i];
                ulong index_i = v_i->ordering_id_;
                ulong dim_i = v_i->dimension_;
                Eigen::MatrixXd JtW;
                if(factor->loss_function_)
                    JtW = jacobia_i.transpose() * factor->robust_information_;
                else
                    JtW = jacobia_i.transpose() * factor->information_;
                for(size_t j = i; j < verticies.size(); ++j)
                {
                    auto v_j = verticies[j];
                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id_;
                    ulong dim_j = v_j->dimension_;

                    Eigen::MatrixXd hessian = JtW*jacobian_j;
                    H.block(index_i,index_j,dim_i,dim_j).noalias() += hessian;
                    if(j != i)
                        H.block(index_j,index_i,dim_j,dim_i).noalias() += hessian.transpose();
                }
                b.segment(index_i,dim_i).noalias() -= JtW * factor->error_;
            }

        }
        Hessian_ = H;
        b_ = b;

        delta_x_ = Eigen::VectorXd::Zero(all_states_size); 
    }

    void setOrdering()
    {
        ulong ordering_id = 0;
        for(auto vertex : vertices_)
        {
            vertex->ordering_id_ = ordering_id;
            ordering_id += vertex->dimension_;
        }
        all_states_size = ordering_id;
    }


    void addFactors(std::shared_ptr<Factor> factor,std::shared_ptr<LossFuntion> loss_function,vector<std::shared_ptr<Vertex>> vs,int dim)
    {
        factors_.push_back(factor);
        factor->setParametersDims(dim,vs);
        factor->setLossFunction(loss_function);
    }

    void addVertex(std::shared_ptr<Vertex> v)
    {
        vertices_.push_back(v);
    }

    vector<std::shared_ptr<Vertex>>  vertices_;    //估计参数
    vector<std::shared_ptr<Factor>>  factors_;  

    Eigen::MatrixXd Hessian_;
    Eigen::MatrixXd b_;
    Eigen::VectorXd delta_x_;
    
    int all_states_size;
    double currentChi_;

};


int main(int argc, char** argv)
{
    double a=0.1,b=0.5,c=2.0;
    int N = 100;
    double w_sigma = 0.05;
    
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.,w_sigma);

    LSSolver solver;

    std::shared_ptr<Vertex> abc(new Vertex(3));
    abc->data_ << 0.,0.,0.;
    solver.addVertex(abc);

    for(int i = 0; i < N; ++i)
    {
        double x = i/100.0;
        double n = noise(generator);
        double y = std::exp(a*x*x+b*x+c)+n;
        Eigen::VectorXd meas(2);
        meas << x,y;

        std::shared_ptr<CurveFittingFactor> factor(new CurveFittingFactor(meas));
        Eigen::MatrixXd information_matrix(1,1);
        information_matrix << 1/(w_sigma*w_sigma);
        factor->setInformationMatrix(information_matrix);
        solver.addFactors(factor,std::shared_ptr<LossFuntion>(new CauchyLoss(0.5)),vector<std::shared_ptr<Vertex>>{abc},1);
//        solver.addFactors(factor,NULL,vector<std::shared_ptr<Vertex>>{abc},1);
    }
    
    //add outlier to verify the performance of adding loss_function
    for(int i = 0; i < 3; ++i)
    {
        double x = i;
        double y = i;
        Eigen::VectorXd meas(2);
        meas << x,y;
        std::shared_ptr<CurveFittingFactor> factor(new CurveFittingFactor(meas));
        Eigen::MatrixXd information_matrix(1,1);
        information_matrix << 1/(w_sigma*w_sigma);
        factor->setInformationMatrix(information_matrix);
        solver.addFactors(factor,std::shared_ptr<LossFuntion>(new CauchyLoss(0.5)),vector<std::shared_ptr<Vertex>>{abc},1);
//        solver.addFactors(factor,NULL,vector<std::shared_ptr<Vertex>>{abc},1);
    }
    
    
    solver.solve(50);

    std::cout << "------After optimization, we got these parameters : " << std:: endl;
    std::cout << abc->data_.transpose() << std::endl;
    std::cout <<"-------ground truth : " << std::endl;
    std::cout << "0.1 0.5 2.0" << std::endl;
    
    std::cout <<"-------estimation error: " << std::endl;
    Eigen::Vector3d true_value;
    true_value << 0.1 ,0.5, 2.0;
    double estimate_error = (abc->data_-true_value).squaredNorm();
    std::cout << estimate_error << std::endl;
    
    return 0;
}
