#ifndef ai_model_HPP
#define ai_model_HPP

#include <torch/script.h>
#include <vector>
#include <string>
#include "karts/abstract_kart.hpp"

class ai_model {
public:
    ai_model();
    Vec3 getTargetAI(const std::vector<float>& data);

    // Functions for reading and scaling/normalising data
    std::tuple<float, float, float> readOutputScalingParameters(const std::string& filePath);
    std::unordered_map<std::string, std::pair<float, float>> readInputNormaliseParameters(const std::string& filePath);
    std::vector<float> parseNumbers(const std::string& str);
    float applyInverseMinMaxScaler(float scaled);
    void normaliseInputData(std::vector<float>& data, const std::unordered_map<std::string, std::pair<float, float>>& parameters);
    Vec3 decodeOutput(torch::Tensor control_output);

    // Functions for adding noise to data
    Vec3 addNoiseToOutputData(const Vec3& output);
    void addNoiseToInputData(std::vector<float>& data, const std::unordered_map<std::string, std::pair<float, float>>& parameters);
    
    // booleans if AI is enabled
    bool inputNoiseBlue = true;
    bool outputNoiseBlue = true;
    bool trainingNoiseBlue = true;
    bool inputNoiseRed = false;
    bool outputNoiseRed = false;
    bool trainingNoiseRed = false;
    bool blueAI = true;
    bool redAI = true;
    bool isSwitch = true;

    // old models
    torch::Tensor run_model(const std::vector<float>& data);
    void lower_actions(KartControl* m_controls, const std::vector<float>& curr_kart_data);
    std::vector<std::vector<float>>& get_lower_actions(const std::vector<float>& data);
    std::vector<float> moves_towards_ball(const std::vector<float>& data);
    std::vector<float> move_forward(const std::vector<float>& data);
    std::vector<float> move_forward_left(const std::vector<float>& data);
    std::vector<float> move_forward_right(const std::vector<float>& data);
    std::vector<float> move_backward(const std::vector<float>& data);
    std::vector<float> move_backward_left(const std::vector<float>& data);
    std::vector<float> move_backward_right(const std::vector<float>& data);

private:
    torch::jit::script::Module model_team_blue;
    torch::jit::script::Module model_team_red;
    torch::jit::script::Module model_team_blue_noise;
    torch::jit::script::Module model_team_red_noise;
    torch::jit::script::Module model_lower_actions;
    float min;
    float max;
};

#endif // ai_model_HPP
