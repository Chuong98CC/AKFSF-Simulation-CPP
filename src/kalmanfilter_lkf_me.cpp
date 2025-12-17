// ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Linear Kalman Filter
//
// ####### STUDENT FILE #######
//
// Usage:
// -Rename this file to "kalmanfilter.cpp" if you want to use this code.

#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr bool INIT_ON_FIRST_PREDICTION = true;
constexpr double INIT_POS_STD = 3.0;
constexpr double INIT_VEL_STD = 10.0;
constexpr double ACCEL_STD = 0.2;
constexpr double GPS_POS_STD = 3.0;
// -------------------------------------------------- //

KalmanFilter::KalmanFilter()
{
    // Initialize the State Transition Matrix (F) and Process Noise Covariance (Q)
    F = Matrix4d::Identity();

    L = Eigen::Matrix<double,4,2>::Zero();

    // Note: The actual values in F and Q will depend on the specific system model
    // and should be set according to the problem requirements.
}
void KalmanFilter::predictionStep(double dt)
{
    if (!isInitialised() && INIT_ON_FIRST_PREDICTION)
    {
        // Implement the State Vector and Covariance Matrix Initialisation in the
        // section below if you want to initialise the filter WITHOUT waiting for
        // the first measurement to occur. Make sure you call the setState() /
        // setCovariance() functions once you have generated the initial conditions.
        // Hint: Assume the state vector has the form [X,Y,VX,VY].
        // Hint: You can use the constants: INIT_POS_STD, INIT_VEL_STD
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
            VectorXd state = Vector4d::Zero();
            MatrixXd cov = Matrix4d::Zero();
            state[2] = 5.0 * std::cos(M_PI_4); // Initial VX
            state[3] = 5.0 * std::sin(M_PI_4); // Initial VY
            // Assume the initial position is (X,Y) = (0,0) m
            // Assume the initial velocity is 5 m/s at 45 degrees (VX,VY) = (5*cos(45deg),5*sin(45deg)) m/s
            cov(0,0) = INIT_POS_STD * INIT_POS_STD; // Position X Variance
            cov(1,1) = INIT_POS_STD * INIT_POS_STD; // Position Y Variance
            cov(2,2) = INIT_VEL_STD * INIT_VEL_STD; // Velocity X Variance
            cov(3,3) = INIT_VEL_STD * INIT_VEL_STD; // Velocity Y Variance
            setState(state);
            setCovariance(cov);
        // ----------------------------------------------------------------------- //
    }

    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Prediction Step for the system in the  
        // section below.
        // Hint: You can use the constants: ACCEL_STD
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
        F(0, 2) = dt;
        F(1, 3) = dt;
        double dt2 = dt * dt;
        L(0,0) = 0.5 * dt2;
        L(1,1) = 0.5 * dt2;
        L(2,0) = dt;
        L(3,1) = dt;
        state = F * state; // State Prediction Step
        cov = F * cov * F.transpose() + (ACCEL_STD * ACCEL_STD) * L * L.transpose(); // Covariance Prediction Step
        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the GPS Measurements in the 
        // section below.
        // Hint: Assume that the GPS sensor has a 3m (1 sigma) position uncertainty.
        // Hint: You can use the constants: GPS_POS_STD
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE 
        Eigen::Matrix<double, 2, 4> H;
        H << 1, 0, 0, 0,
             0, 1, 0, 0;
        Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (GPS_POS_STD * GPS_POS_STD);
        Eigen::Vector2d z;
        z << meas.x, meas.y;
        Eigen::Vector2d y = z - H * state;
        Eigen::Matrix2d S = H * cov * H.transpose() + R;
        // Theory
        // Eigen::MatrixXd K = cov * H.transpose() * S.inverse(); (1)
        // cov = (Id - K * H) * cov;

        // More efficient approach, 
        // rearranged (1) as: K^T = S^{-1}^T * (P * H^T)^T = S^{-1} * B
        // So: K = (S^{-1} * B)^T, with S is symmetric
        // Matrix< 2, 4> B = (P * H^T)^T= H * P^T;
        // Because the special structure of update_mat H=[I2x2 02x2], we can simplify B matrix
        Eigen::Matrix<double, 2, 4> B = cov.transpose().topRows<2>();
        Eigen::Matrix<double, 4, 2> K = (S.ldlt().solve(B)).transpose();
        state = state + K * y;
        cov -=  K * (S)*K.transpose();

        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // Implement the State Vector and Covariance Matrix Initialisation in the
        // section below. Make sure you call the setState/setCovariance functions
        // once you have generated the initial conditions.
        // Hint: Assume the state vector has the form [X,Y,VX,VY].
        // Hint: You can use the constants: GPS_POS_STD, INIT_VEL_STD
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
            VectorXd state = Vector4d::Zero();
            MatrixXd cov = Matrix4d::Zero();


            setState(state);
            setCovariance(cov);
        // ----------------------------------------------------------------------- //
    }        
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance()
{
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0){pos_cov << cov(0,0), cov(0,1), cov(1,0), cov(1,1);}
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState()
{
    if (isInitialised())
    {
        VectorXd state = getState(); // STATE VECTOR [X,Y,VX,VY]
        double psi = std::atan2(state[3],state[2]);
        double V = std::sqrt(state[2]*state[2] + state[3]*state[3]);
        return VehicleState(state[0],state[1],psi,V);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt){predictionStep(dt);}
void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map){}
void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map){}

