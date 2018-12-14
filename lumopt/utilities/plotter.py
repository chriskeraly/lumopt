import matplotlib as mpl

# #mpl.use('TkAgg')
 #mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


class Plotter(object):
    '''
    This class deals with plotting various information regarding the optimization happening
    '''
    def __init__(self,plotting_options=None,movie=True):
        '''
        :param plotting_options: A dictionnary of the plotting options
        '''

        self.fig, self.ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
        self.fig.show()
        self.movie=movie
        if movie:
            try:
                FFMpegWriter = manimation.writers['ffmpeg']
                metadata = dict(title='Optimization', artist='LumOpt',
                                comment='Continuous Adjoint Based Optimization')
                self.writer = FFMpegWriter(fps=2, metadata=metadata)
                #self.writer_saving=self.writer.saving(self.fig, "optimization.mp4", 100)

            except:
                print('Cannot make movie')
                self.movie=False


    def update(self,optimization):
        optimization.optimizer.plot(fomax=self.ax[0,0],paramsax=self.ax[0,1],gradients_ax=self.ax[0,2])

        from lumopt.optimization import Optimization
        if isinstance(optimization,Optimization):
            if not optimization.geometry.plot(self.ax[1,0]):
                optimization.gradient_fields.plot_eps(self.ax[1,0])
            optimization.gradient_fields.plot(self.fig,self.ax[1,1],self.ax[1,2])
            # optimization.geometry.plot(self.ax[1,0])
        else:
            if not optimization.optimizations[0].geometry.plot(self.ax[1,0]):
                try:
                    optimization.optimizations[0].gradient_fields.plot_eps(self.ax[1,0])
                except:
                    print("can't plot geometry")
            try:
                optimization.optimizations[0].gradient_fields.plot(self.fig,self.ax[1,1],self.ax[1,2])
            except:
                print("can't plot gradient fields")
            # optimization.optimizations[0].geometry.plot(self.ax[1,0])
        print('plot updated')
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.movie:
            #self.fig.savefig(self.writer._frame_sink(),format=self.writer.frame_format,dpi=100)
            self.writer.grab_frame()
            print('Saved frame')


