run using:
    python robotvision.py -m model.model -l lb.pickle 

required args:
    -m      path to model file
    -l      path to label pickle file

optional args:
    -c      float; min confidence to label picture      default=.7
    -t      int; thickness of the crosshair in pixels   default=4
    -wc     int; bigger # => wider crosshair            default=3
    -hc     int; bigger # => taller crosshair           default=4
    -o      str; path for output video                  default=result.avi

model info:
    -acc = 98.64
    -validation_acc=95.4
    -# human_pics=213
    -# robot pics=401
    -graph of training=trainplot.png
