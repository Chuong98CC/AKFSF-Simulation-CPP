// ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Extended Kalman Filter
//
// ####### STUDENT FILE #######
//
// Usage:
// -Rename this file to "kalmanfilter.cpp" if you want to use this code.

#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr double ACCEL_STD = 1.0;
constexpr double GYRO_STD = 0.01/180.0 * M_PI;
constexpr double GYRO_BIAS_STD = 0.005/180.0 * M_PI; // Gyro Bias Process Model Noise (CAPSTONE)
constexpr double INIT_VEL_STD = 10.0;
constexpr double INIT_PSI_STD = 45.0/180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;

constexpr int num_state = 5; // [px, py, psi, v, gyro_bias]
constexpr int num_meas_lidar = 2;
constexpr int num_meas_gps = 2;
// -------------------------------------------------- //

// ----------------------------------------------------------------------- //
// USEFUL HELPER FUNCTIONS
VectorXd normaliseState(VectorXd state)
{
    state(2) = wrapAngle(state(2));
    return state;
}
VectorXd normaliseLidarMeasurement(VectorXd meas)
{
    meas(1) = wrapAngle(meas(1));
    return meas;
}
// ----------------------------------------------------------------------- //

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the Lidar Measurements in the 
        // section below.
        // HINT: Use the normaliseState() and normaliseLidarMeasurement() functions
        // to always keep angle values within correct range.
        // HINT: Do not normalise during sigma point calculation!
        // HINT: You can use the constants: LIDAR_RANGE_STD, LIDAR_THETA_STD
        // HINT: The mapped-matched beacon position can be accessed by the variables
        // map_beacon.x and map_beacon.y
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
        double px = state(0);
        double py = state(1);
        double psi = state(2);
        BeaconData map_beacon;
        // ----------------------------------------------------------------------- //
        // Capstone Addition: If no beacon match found, try nearest neighbour within reasonable range
        if (meas.id == -1){
            // No Beacon Match Found - Try Nearest Neighbour within Reasonable Range
            double lidar_r = meas.range;
            double lidar_bearing = meas.theta;
            double expected_Lx = px + lidar_r*cos(wrapAngle(psi+lidar_bearing));
            double expected_Ly = py + lidar_r*sin(wrapAngle(psi+lidar_bearing));
            std::vector<BeaconData> nearby_beacons = map.getBeaconsWithinRange(expected_Lx, expected_Ly, 6*LIDAR_RANGE_STD); 
            int num_beacons = nearby_beacons.size();
            if (num_beacons>0)
            {
                // Choose the closest beacon
                map_beacon = nearby_beacons[0];  // Assign to outer variable, not create new one
                double dx = expected_Lx - map_beacon.x;
                double dy = expected_Ly - map_beacon.y;
                double closest_dist = sqrt(dx*dx + dy*dy);
                for (int i=1; i<num_beacons; ++i)
                {
                    dx = expected_Lx - nearby_beacons[i].x;
                    dy = expected_Ly - nearby_beacons[i].y;
                    double dist = sqrt(dx*dx + dy*dy);
                    if (dist<closest_dist)
                    {
                        closest_dist = dist;
                        map_beacon = nearby_beacons[i]; 
                    }   
                }
                
                std::cout << " - Nearest Beacon ID " << map_beacon.id << " Chosen at Distance " << closest_dist << std::endl;
                meas.id = map_beacon.id; // Update measurement id to matched beacon
            }  
            else {
                std::cout << " - No Nearby Beacon Found for Lidar Measurement!" << std::endl;
                return; // No nearby beacon found, cannot update
            }  
        }
        else {
            map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        }

        if (map_beacon.id == -1) {
            std::cout << "Beacon ID " << meas.id << " not found in map!" << std::endl;
            return;
        }
        // predict the measurement using the prior state and the process model
        // Note: dx, dy are beacon relative to vehicle for correct bearing
        double dx = map_beacon.x - px;
        double dy = map_beacon.y - py;
        double r2 =  dx*dx + dy*dy;
        double r_hat = sqrt(r2);                         
        double theta_hat = wrapAngle(atan2(dy, dx) - psi);
        Vector2d z_hat;
        z_hat << r_hat, theta_hat;

        //  Compute Jacobian H (derivatives of measurement model w.r.t. state)
        //  z = [r, theta] = [sqrt((bx-px)^2+(by-py)^2), atan2(by-py, bx-px) - psi]
        //  dz/d[px,py,psi,v,bias]
        Eigen::Matrix<double, 2, num_state> H;
        H << -dx/r_hat, -dy/r_hat, 0, 0, 0,
                dy/r2, -dx/r2, -1, 0, 0;
        
        // Compute measurement covariance R
        Matrix2d R = Matrix2d::Zero();
        R(0,0) = LIDAR_RANGE_STD*LIDAR_RANGE_STD;
        R(1,1) = LIDAR_THETA_STD*LIDAR_THETA_STD;
        // Compute Sensor Covariance S
        Matrix2d S = H * cov * H.transpose() + R;
        
        // Compute Kalman Gain K
        MatrixXd K = cov * H.transpose() * S.inverse();

        // Update State and Covariance
        // compute the measurement residual
        Vector2d z;
        z << meas.range, meas.theta;
        Vector2d y = z - z_hat;
        y(1) = wrapAngle(y(1));  // Wrap angle innovation
        state = state + K * y;
        state = normaliseState(state);  // Normalize heading after update
        cov = (MatrixXd::Identity(num_state,num_state) - K * H) * cov;
        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Prediction Step for the system in the  
        // section below.
        // HINT: Assume the state vector has the form [PX, PY, PSI, V].
        // HINT: Use the Gyroscope measurement as an input into the prediction step.
        // HINT: You can use the constants: ACCEL_STD, GYRO_STD
        // HINT: Use the normaliseState() function to always keep angle values within correct range.
        // HINT: Do NOT normalise during sigma point calculation!
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
        double psi = state(2);
        double V = state(3);
        double gyro_bias = state(4);
        double psi_dot = gyro.psi_dot;

        // Prediction state
        VectorXd input = VectorXd::Zero(num_state);
        input << V*cos(psi), V*sin(psi), psi_dot - gyro_bias, 0, 0;
        state += input*dt;
        state = normaliseState(state);

        // Prediction covariance
        MatrixXd Q = MatrixXd::Zero(num_state,num_state);
        Q(2,2) = dt*dt*GYRO_STD*GYRO_STD;  
        Q(3,3) = dt*dt*ACCEL_STD*ACCEL_STD;
        Q(4,4) = dt*dt*GYRO_BIAS_STD*GYRO_BIAS_STD;  // process noise for gyro bias

        MatrixXd F = MatrixXd::Identity(num_state,num_state);
        F(0,2) = -V*sin(psi)*dt;
        F(0,3) = cos(psi)*dt;
        F(1,2) = V*cos(psi)*dt;
        F(1,3) = sin(psi)*dt;
        F(2,4) = -dt; // added for gyro bias
        cov = F*cov*F.transpose() + Q;
        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    } 
    else{
        // Assume Driving Straight or Stationary to Estimate Gyro Bias (CAPSTONE)
        m_init_bias = gyro.psi_dot;
        m_init_bias_valid = true;
        std::cout << "INIT<BIAS> @ " << m_init_bias << std::endl;
    }
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    // All this code is the same as the LKF as the measurement model is linear
    // so the UKF update state would just produce the same result.
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();
    
        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd::Zero(num_meas_gps,num_state);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H(0,0) = 1;
        H(1,1) = 1;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd z_error = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

         // GPS Innovation Check (NIS) Dealing with Fault Values
        VectorXd NIS = z_error.transpose()*S.inverse()*z_error;
        if (NIS(0) < 100) // Put a threshold here
        { 
            state = state + K*z_error;
            cov = (MatrixXd::Identity(num_state,num_state) - K*H) * cov;
            setState(state);
            setCovariance(cov);
        } else
        {
            std::cout << "GPS Measurement Rejected by NIS Check: " << NIS(0) << std::endl;
        }
        
    }
    else
    {
        // You may modify this initialisation routine if you can think of a more
        // robust and accuracy way of initialising the filter.
        // ----------------------------------------------------------------------- //
        // YOU ARE FREE TO MODIFY THE FOLLOWING CODE HERE
        // initialize velocity to zero
        
        if (!m_init_position_valid)
        {
            m_init_position_x = meas.x;
            m_init_position_y = meas.y;
            m_init_position_valid = true;
            std::cout << "INIT<POSITION> @ " << m_init_position_x << "," << m_init_position_y << std::endl;
            m_init_velocity = 0;
            m_init_velocity_valid = true;
            std::cout << "INIT<VELOCITY> @ " << m_init_velocity << std::endl;
        }
        else if (!m_init_heading_valid )
        {
            double delta_x = meas.x - m_init_position_x;
            double delta_y = meas.y - m_init_position_y;
            // Check If We have moved Enough
            if ((delta_x*delta_x + delta_y*delta_y) > 3*GPS_POS_STD)
            {
                // Estimate Heading from
                m_init_heading = atan2(delta_y,delta_x);
                m_init_heading_valid = true;
                std::cout << "INIT<HEADING> by GPS @ " << m_init_heading << std::endl;
            }
        }
        
        if (m_init_position_valid &&
            m_init_velocity_valid &&
            m_init_heading_valid &&
            m_init_bias_valid)
        {
            VectorXd state = VectorXd::Zero(num_state);
            MatrixXd cov = MatrixXd::Zero(num_state,num_state);

            state(0) = m_init_position_x;
            state(1) = m_init_position_y;
            state(2) = m_init_heading; 
            state(3) = m_init_velocity;
            state(4) = m_init_bias; // Initial Gyro Bias 
            cov(0,0) = GPS_POS_STD*GPS_POS_STD;
            cov(1,1) = GPS_POS_STD*GPS_POS_STD;
            cov(2,2) = INIT_PSI_STD*INIT_PSI_STD;
            cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;
            cov(4,4) = GYRO_STD*GYRO_STD; // Initial Gyro Bias Uncertainty (use GYRO_STD for reasonable estimate)
            std::cout << "FILTER Fully INIT" << std::endl;
            setState(state);
            setCovariance(cov);
        }

        // ----------------------------------------------------------------------- //
    }             
}

void KalmanFilter::init_heading_from_lidar(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    if (!m_init_position_valid){
        std::cout << "LIDAR Measurement Received before GPS Init - Ignored" << std::endl;
        return;
    }
    std::vector<double> psi_candidates;
    for (const auto& meas : dataset) {
        if (meas.id != -1) {
            BeaconData assoc_beacon = map.getBeaconWithId(meas.id);
            double dx = assoc_beacon.x - m_init_position_x;
            double dy = assoc_beacon.y - m_init_position_y;
            double expected_bearing = atan2(dy, dx);
            double psi = wrapAngle(expected_bearing - meas.theta);
            psi_candidates.push_back(psi);
        }
    }

    if (!psi_candidates.empty()) {
        // average the candidate headings through circular mean to handle angle wrapping
        double sum_sin = 0.0;
        double sum_cos = 0.0;
        for (double angle : psi_candidates) {
            sum_sin += sin(angle);
            sum_cos += cos(angle);
        }
        m_init_heading = wrapAngle(atan2(sum_sin, sum_cos));
        m_init_heading_valid = true;
        std::cout << "INIT<HEADING> from LIDAR @ " << m_init_heading << std::endl;
    }
}

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    auto meas = dataset[0];
    if (meas.id != -1) {  
        // for simulation that already provides data association id directly
        if (!m_init_heading_valid) { init_heading_from_lidar(dataset, map); }
    }
    for(const auto& meas : dataset) { handleLidarMeasurement(meas, map); }
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
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0],state[1],state[2],state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt){}
