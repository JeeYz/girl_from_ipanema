from pygame_modules.setting_builder.SettingBuilder import BasicSettingBuilder

if __name__ == "__main__":
    app_main = BasicSettingBuilder()\
                .set_basic_width(600)\
                .set_basic_height(480)\
                .set_size()\
                .set_background_color((0, 0, 0))\
                .set_fps_time(30)\
                .set_up_basic()

    app_main.do_execute()
            
    
