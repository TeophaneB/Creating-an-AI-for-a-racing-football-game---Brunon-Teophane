#include "ai_model.hpp"
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

//-----------------------------------------------------------------------------
ai_model::ai_model() {
    const char* path_red = R"(/media/sf_Y3/Game/PT_model/models/net_0_diff.pt)";
    const char* path_blue = R"(/media/sf_Y3/Game/PT_model/models/net_1_diff.pt)";
    const char* path_red_noise = R"(/media/sf_Y3/Game/PT_model/models/net_0_diff_noise.pt)";
    const char* path_blue_noise = R"(/media/sf_Y3/Game/PT_model/models/net_1_diff_noise.pt)";
    const char* path_lower_actions = R"(/media/sf_Y3/Game/PT_model/models/net.pt)";
    model_team_red = torch::jit::load(path_red);
    model_team_blue = torch::jit::load(path_blue);
    model_team_red_noise = torch::jit::load(path_red_noise);
    model_team_blue_noise = torch::jit::load(path_blue_noise);
    model_lower_actions = torch::jit::load(path_lower_actions);
    auto scalingParameters = readOutputScalingParameters("/media/sf_Y3/Game/PT_model/scaling_parameters.txt");
    min = std::get<0>(scalingParameters);
    max = std::get<1>(scalingParameters);
    std::cout << "Model loaded successfully" << std::endl;
}

//-----------------------------------------------------------------------------
void debugPrint(const std::vector<float>& data) {   // helper fct used for testing
	std::cout << "Data: ";
	for (const auto& value : data) {
		std::cout << value << " ";
	}
	std::cout << std::endl;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::parseNumbers(const std::string& str) {     // parsing helper function
    std::vector<float> numbers;
    std::istringstream iss(str);
    float num;
    while (iss >> num) {
        numbers.push_back(num);
        // Skip commas
        iss.ignore();
    }
    return numbers;
}

//-----------------------------------------------------------------------------
std::unordered_map<std::string, std::pair<float, float>> ai_model::readInputNormaliseParameters(const std::string& filePath) {    // open file and read normalising parameters for input
    //std::cout << "ai_model::readInputNormaliseParameters" << std::endl;
    std::unordered_map<std::string, std::pair<float, float>> parameters;
    std::ifstream file(filePath);
    std::string line;

    // Find means
    size_t meansStart = line.find("[") + 1;
    size_t meansEnd = line.find("]", meansStart);
    std::string meansStr = line.substr(meansStart, meansEnd - meansStart);

    // Find scales
    size_t scalesStart = line.find("[", meansEnd) + 1;
    size_t scalesEnd = line.find("]", scalesStart);
    std::string scalesStr = line.substr(scalesStart, scalesEnd - scalesStart);

    std::vector<float> means = parseNumbers(meansStr);
    std::vector<float> scales = parseNumbers(scalesStr);

    for (size_t i = 0; i < means.size(); ++i) {     // turn the means and scales into a map
        parameters[std::to_string(i)] = { means[i], scales[i] };
    }

    //std::cout << "Number of parameters loaded: " << parameters.size() << std::endl;

    return parameters;
}

//-----------------------------------------------------------------------------
void ai_model::normaliseInputData(std::vector<float>& data, const std::unordered_map<std::string, std::pair<float, float>>& parameters) {   // scale the input data
    for (size_t i = 0; i < data.size(); ++i) {
        std::string key = std::to_string(i);
        auto it = parameters.find(key); // get normalising parameters
        if (it != parameters.end()) {
            data[i] = (data[i] - it->second.first) / it->second.second;     // normalise input
        }
    }
}

//-----------------------------------------------------------------------------
std::tuple<float, float, float> ai_model::readOutputScalingParameters(const std::string& filePath) {	// read scaling parameters for output
    //std::cout << "ai_model::readOutputScalingParameters" << std::endl;
    std::ifstream file(filePath);
    std::string line;
    float min = 0.0f, max = 0.0f, scale = 0.0f;

    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string tempMin, tempMax, tempScale;
        char comma1, comma2;

        if (iss >> tempMin >> min >> comma1 >> tempMax >> max >> comma2 >> tempScale >> scale) {
            if (tempMin != "Min:" || tempMax != "Max:" || tempScale != "Scale:" || comma1 != ',' || comma2 != ',') {
                std::cerr << "Unexpected format in scaling parameters file." << std::endl;
                throw std::runtime_error("Unexpected format in scaling parameters file.");
            }
        }
    }

    return { min, max, scale };
}

//-----------------------------------------------------------------------------
float ai_model::applyInverseMinMaxScaler(float scaled) {    // reverse the scaling formula
    return scaled * (max - min) + min;      // use the orignal scaling parameters
}

//-----------------------------------------------------------------------------
Vec3 ai_model::decodeOutput(torch::Tensor control_output) {   // decode the output by reversing the scale
    //std::cout << "Scaled control output: " << control_output[0].item<float>() << ", " << control_output[1].item<float>() << ", " << control_output[2].item<float>() << std::endl;

    float x = applyInverseMinMaxScaler(control_output[0].item<float>());
    float y = applyInverseMinMaxScaler(control_output[1].item<float>());
    float z = applyInverseMinMaxScaler(control_output[2].item<float>());

    //std::cout << "Unscaled control output: " << x << ", " << y << ", " << z << std::endl;

    return Vec3(x, y, z);
}

//-----------------------------------------------------------------------------
void ai_model::addNoiseToInputData(std::vector<float>& data, const std::unordered_map<std::string, std::pair<float, float>>& parameters) {
    //std::wcout << "ai_model::addNoiseToInputData" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd()); // Random number generator

    for (size_t i = 0; i < data.size(); ++i) {
        std::string key = std::to_string(i);
        auto it = parameters.find(key); // get normalising parameters
        if (it != parameters.end()) {
            float mean = it->second.first;
            float scale = it->second.second;
            std::normal_distribution<> distr(mean, scale); // Using normal distribution centered around mean with variance
            data[i] += distr(gen);
        }
    }
}

//-----------------------------------------------------------------------------
Vec3 ai_model::addNoiseToOutputData(const Vec3& output) {
    //std::wcout << "ai_model::addNoiseToOutputData" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd()); // Random number generator
    std::uniform_real_distribution<> distr(min, max); // Uniform distribution within the min-max range

    float noise_intensity = 0.1f; 
    float x = output.getX() + distr(gen) * (max - min) * noise_intensity;   
    float y = output.getY() + distr(gen) * (max - min) * noise_intensity;
    float z = output.getZ() + distr(gen) * (max - min) * noise_intensity;

    return Vec3(x, y, z);
}

//-----------------------------------------------------------------------------
Vec3 ai_model::getTargetAI(const std::vector<float>& data) {
    //std::cout << "ai_model::getTargetAI" << std::endl;

    // Ready data (without scaling)
    std::vector<float> model_ready_data;
    model_ready_data.insert(model_ready_data.end(), data.begin() + 1, data.end() - 3); // remove kart id && target values

    // Run model
    torch::Tensor control_output;
    Vec3 target_point;
    if (data.at(0) == 1) {  // Blue team
        //std::cout << "- Blue" << std::endl;
        auto scalingParameters = readInputNormaliseParameters("/media/sf_Y3/Game/PT_model/scaling_parameters_team_1.txt");

        if (inputNoiseBlue) {           // add noise to input data
            //std::cout << "input noise added" << std::endl;
            //std::cout << "before add noise:" << std::endl;
            //debugPrint(model_ready_data);
            addNoiseToInputData(model_ready_data, scalingParameters);
            //std::cout << "after add noise:" << std::endl;
            //debugPrint(model_ready_data);
        }
            
                                        // sacle the input the same as the training phase
        normaliseInputData(model_ready_data, scalingParameters);

        if (trainingNoiseBlue) {        // choose model that trained on noise
            //std::cout << "traning noise added" << std::endl;
			control_output = model_team_blue_noise.forward({ torch::tensor(model_ready_data) }).toTensor();
		}
		else {                          // normal model
			control_output = model_team_blue.forward({ torch::tensor(model_ready_data) }).toTensor();
		}

                                        // decode the scale output
        target_point = decodeOutput(control_output);

        if (outputNoiseBlue) {          // add noise to output data
            //std::cout << "output noise added" << std::endl;
            return addNoiseToOutputData(target_point);
        }
    }
    else if (data.at(0) == 0) { // Red team (same code as blue team)
        //std::cout << "- Red" << std::endl;
        auto scalingParameters = readInputNormaliseParameters("/media/sf_Y3/Game/PT_model/scaling_parameters_team_0.txt");

        if (inputNoiseRed) {
            //std::cout << "input noise added" << std::endl;
            //std::cout << "before add noise:" << std::endl;
            //debugPrint(model_ready_data);
            addNoiseToInputData(model_ready_data, scalingParameters);
            //std::cout << "after add noise:" << std::endl;
            //debugPrint(model_ready_data);
        }

        normaliseInputData(model_ready_data, scalingParameters);
        
        if (trainingNoiseRed) {
            //std::cout << "traning noise added" << std::endl;
            control_output = model_team_red_noise.forward({ torch::tensor(model_ready_data) }).toTensor();
        }
        else {
			control_output = model_team_red.forward({ torch::tensor(model_ready_data) }).toTensor();
		}

        target_point = decodeOutput(control_output);

        if (outputNoiseRed) {
            //std::cout << "output noise added" << std::endl;
            return addNoiseToOutputData(target_point);
        }
    }
    
    return target_point;
}






                            /// OLD MODELS ///

//-----------------------------------------------------------------------------
torch::Tensor ai_model::run_model(const std::vector<float>& data) {
    torch::Tensor data_tensor = torch::from_blob((float*)data.data(), { static_cast<long>(data.size()) }, torch::kFloat32);

    torch::NoGradGuard no_grad;
    torch::Tensor control_output;
    try {
        control_output = model_lower_actions.forward({ data_tensor }).toTensor();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error running the model: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return control_output;

}

//-----------------------------------------------------------------------------
void ai_model::lower_actions(KartControl* m_controls, const std::vector<float>& curr_kart_data) {
    std::cout << "ai_model::lower_actions" << std::endl;
    const std::vector<std::vector<float>>& moves = get_lower_actions(curr_kart_data);
    float best_score = -std::numeric_limits<float>::infinity();
    std::vector<float> best_move;

    for (const auto& move : moves) {
        torch::Tensor control_output = run_model(move);
        float score = control_output.item<float>();
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }

    m_controls->setAccel(best_move[4]);
    m_controls->setSteer(best_move[5]);
    m_controls->setBrake(best_move[10]);
    m_controls->setNitro(best_move[11]);

}

//-----------------------------------------------------------------------------
std::vector<std::vector<float>>& ai_model::get_lower_actions(const std::vector<float>& data) {
    static std::vector<std::vector<float>> moves_array;

    // Altering the current data into possible actions
    moves_array.push_back(moves_towards_ball(data));
    moves_array.push_back(move_forward(data));
    moves_array.push_back(move_forward_left(data));
    moves_array.push_back(move_forward_right(data));
    moves_array.push_back(move_backward(data));
    moves_array.push_back(move_backward_left(data));
    moves_array.push_back(move_backward_right(data));

    return moves_array;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::moves_towards_ball(const std::vector<float>& data) {
    std::vector<float> new_data(data);
    float ball_x = data[0];
    float ball_y = data[1];
    float kart_x = data[2];
    float kart_y = data[3];

    // Calculate direction vector towards the ball
    float direction_x = ball_x - kart_x;
    float direction_y = ball_y - kart_y;

    // Normalize direction vector
    float length = std::sqrt(direction_x * direction_x + direction_y * direction_y);
    direction_x /= length;
    direction_y /= length;

    // Apply movement to new_data
    new_data[2] += direction_x;
    new_data[3] += direction_y;

    return new_data;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::move_forward(const std::vector<float>& data) {
    std::vector<float> new_data(data);
    new_data[4] = 1.0;
    new_data[5] = 0.0f;
    new_data[10] = false;
    new_data[11] = true;
    return new_data;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::move_forward_left(const std::vector<float>& data) {
    std::vector<float> new_data(data);
    new_data[4] = 1.0f;
    new_data[5] -= 1.0f;
    new_data[10] = false;
    new_data[11] = false;
    return new_data;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::move_forward_right(const std::vector<float>& data) {
    std::vector<float> new_data(data);
    new_data[4] = 1.0f;
    new_data[5] += 1.0f;
    new_data[10] = false;
    new_data[11] = false;
    return new_data;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::move_backward(const std::vector<float>& data) {
    std::vector<float> new_data(data);
    new_data[4] = 0.0f;
    new_data[5] += 1.0f;
    new_data[10] = false;
    new_data[11] = false;
    return new_data;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::move_backward_right(const std::vector<float>& data) {
    std::vector<float> new_data(data);
    new_data[4] = 0.0f;
    new_data[5] -= 1.0f;
    new_data[10] = false;
    new_data[11] = false;
    return new_data;
}

//-----------------------------------------------------------------------------
std::vector<float> ai_model::move_backward_left(const std::vector<float>& data) {
    std::vector<float> new_data(data);
    new_data[4] = 0.0f;
    new_data[5] += 1.0f;
    new_data[10] = false;
    new_data[11] = false;
    return new_data;
}



