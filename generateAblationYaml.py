import yaml

for ablation in range(1, 11):
    if ablation != 0:
        with open('yamls/qwen2vl_full_uibert_query_summary_polished+aitz_best_sft.yaml', 'r') as file:
            config = yaml.safe_load(file)

        config['dataset'] = f'aitz_action_predict_ablation_{ablation}' 
        config['output_dir'] = f'/data3/trainedModels/qwen2vl-7b/full/uibert_query_summary_polished+aitz_ablation_{ablation}_sft'

        with open(f'./ablationYamls/qwen2vl_full_uibert_query_summary_polished+aitz_ablation_{ablation}_sft.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        print(f'save to ./ablationYamls/qwen2vl_full_uibert_query_summary_polished+aitz_ablation_{ablation}_sft.yaml')
        ##############################################################
        
        with open('yamls/qwen2vl_full_uibert_grounding_polished+aitz_best_sft.yaml', 'r') as file:
            config = yaml.safe_load(file)

        config['dataset'] = f'aitz_action_predict_ablation_{ablation}'  
        config['output_dir'] = f'/data3/trainedModels/qwen2vl-7b/full/uibert_grounding_polished+aitz_ablation_{ablation}_sft'  
        with open(f'./ablationYamls/qwen2vl_full_uibert_grounding_polished+aitz_ablation_{ablation}_sft.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        print(f'save to ./ablationYamls/qwen2vl_full_uibert_grounding_polished+aitz_ablation_{ablation}_sft.yaml')
        ##############################################################
        
        with open('yamls/qwen2vl_full_aitz_best_sft.yaml', 'r') as file:
            config = yaml.safe_load(file)

        config['dataset'] = f'aitz_action_predict_ablation_{ablation}'  
        config['output_dir'] = f'/data3/trainedModels/qwen2vl-7b/full/aitz_ablation_{ablation}_sft'  
        
        with open(f'./ablationYamls/qwen2vl_full_aitz_ablation_{ablation}_sft.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        print(f'save to ./ablationYamls/qwen2vl_full_aitz_ablation_{ablation}_sft.yaml')
        
        
        ############################################
        with open('yamls/qwen2vl_full_aitz_best_sft.yaml', 'r') as file:
            config = yaml.safe_load(file)

        config['dataset'] = f'aitz_action_predict_ablation_{ablation}'  
        config["model_name_or_path"] = "/data4/trainedModels/qwen2vl-7b/full/uibert_grounding_polished+uibert_query_summary_polished"
        config['output_dir'] = f'/data3/trainedModels/qwen2vl-7b/full/uibert_grounding_polished+uibert_query_summary_polished+aitz_ablation_{ablation}_sft'  

        with open(f'./ablationYamls/qwen2vl_full_uibert_grounding_polished+uibert_query_summary_polished+aitz_ablation_{ablation}_sft.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        print(f'save to ./ablationYamls/qwen2vl_full_uibert_grounding_polished+uibert_query_summary_polished+aitz_ablation_{ablation}_sft.yaml')
        
        
        ############################################
        with open('yamls/qwen2vl_full_aitz_best_sft.yaml', 'r') as file:
            config = yaml.safe_load(file)

        config['dataset'] = f'aitz_action_predict_ablation_{ablation}'  
        config["model_name_or_path"] = "/data4/trainedModels/qwen2vl-7b/full/uibert_grounding_polished+android_control_query_summary_polished_epoch3/"
        config['output_dir'] = f'/data3/trainedModels/qwen2vl-7b/full/uibert_grounding_polished+android_control_query_summary_polished_epoch3+aitz_ablation_{ablation}_sft'  
        with open(f'./ablationYamls/qwen2vl_full_uibert_grounding_polished+android_control_query_summary_polished_epoch3+aitz_ablation_{ablation}_sft.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        print(f'save to ./ablationYamls/qwen2vl_full_uibert_grounding_polished+android_control_query_summary_polished_epoch3+aitz_ablation_{ablation}_sft.yaml')
        
        
        ############################################
        with open('yamls/qwen2vl_full_aitz_best_sft.yaml', 'r') as file:
            config = yaml.safe_load(file)

        
        config['dataset'] = f'aitz_action_predict_ablation_{ablation}'  
        config["model_name_or_path"] = "/data3/trainedModels/qwen2vl-7b/full/uibert_grounding+query_summary_polished"
        config['output_dir'] = f'/data3/trainedModels/qwen2vl-7b/full/uibert_grounding+query_summary_polished+aitz_ablation_{ablation}_sft'  
        
        with open(f'./ablationYamls/qwen2vl_full_uibert_grounding+query_summary_polished+aitz_ablation_{ablation}_sft.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        print(f'save to ./ablationYamls/qwen2vl_uibert_grounding+query_summary_polished+aitz_ablation_{ablation}_sft.yaml')
        
        