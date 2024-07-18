import torch
# Le batch size doit varier ainsi le réseau doit rester le mm
class Hilbert2D(torch.nn.Module):
    def __init__(self, transformation: str, mask_type: str, axis=0):
        """
        Initialize a Hilbert2D transformation module.

        Args:
            transformation (str): The type of transformation ("CC","SO","BC","DC") to perform
            mask_type (str): The type of mask ("HT") to use.
            axis (int): The axis to apply the transformation along.
        """
        super(Hilbert2D, self).__init__()
        self.transformation = transformation
        self.mask_type = mask_type
        self.axis = axis
        self.img_shape = None
        self.X = None
        self.Y = None

    def build(self) -> None:
        """
        Build the meshgrid for the transformation.

        Returns:
            None
        """
        self.X, self.Y = torch.meshgrid(torch.fft.fftfreq(self.img_shape[-2]), torch.fft.fftfreq(self.img_shape[-1]),
                                        indexing="ij")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
       Apply the Hilbert transformation to an input image.

       Args:
           img (torch.Tensor): The input image.

       Returns:
           torch.Tensor: The transformed image.
        """
        if self.img_shape is None:
            self.img_shape = img.size()
            self.build()
        return torch.fft.ifft2(self.transformation_mask() * torch.fft.fft2(img))

    @staticmethod
    def sign(Z: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute the sign function on a meshgrid.

        Args:
            Z (torch.Tensor): The meshgrid.
            n (int): One element from the image shape.

        Returns:
            torch.Tensor: The sign values.
        """
        return ((0 < Z) & (Z <= n / 2 - 1)).float() + -((n / 2 + 1 <= Z) & (Z <= n - 1)).float()

    def transformation_mask(self):
        """
        Compute the transformation mask based on the specified transformation and mask type.

        Returns:
            torch.Tensor: The transformation mask.
        """

        if self.transformation == "CC":
            common_component_cc = (1. if self.axis == 0 else -1.) * self.sign(self.X, self.img_shape[-1]) + self.sign(
                self.Y, self.img_shape[-2])
            return -0.5j * common_component_cc if self.mask_type == "HT" else 1 + 0.5 * common_component_cc

        elif self.transformation == "SO":
            X_sign = self.sign(self.X, self.img_shape[-2])
            Y_sign = self.sign(self.Y, self.img_shape[-1])
            common_component_so = (1 + X_sign) * (1 + Y_sign)
            return common_component_so - 1 if self.mask_type == "HT" else common_component_so

        elif self.transformation == "BC":
            X_sign = self.sign(self.X, self.img_shape[-2])
            Y_sign = self.sign(self.Y, self.img_shape[-1])
            common_component_bc = -X_sign * Y_sign
            return common_component_bc if self.mask_type == "HT" else (1 + 1j * common_component_bc)

        elif self.transformation == "DC":
            common_component_dc = self.sign(self.X, self.img_shape[-1]) if self.axis == 0 else self.sign(self.Y,
                                                                                                         self.img_shape[
                                                                                                             -2])
            return -1j * common_component_dc if self.mask_type == "HT" else 1 + common_component_dc


class WeightedHilbert2D(torch.nn.Module):
    def __init__(self):
        super(WeightedHilbert2D, self).__init__()
        self.img_shape = None
        self.X = None
        self.Y = None
        self.img_3D = None
        self.lazy = None
    def build(self,img):
        self.img_shape = [img.size(-3),img.size(-2),img.size(-1)]
        if len(img.size()) > 3: self.img_3D = True
        else: self.img_3D = False
    def forward(self, img):
        if self.lazy is None:
            self.build(img)
            self.lazy = True
        batch_size = img.shape[0]
        X, Y = torch.meshgrid(torch.fft.fftfreq(self.img_shape[-2]), torch.fft.fftfreq(self.img_shape[-1]),
                              indexing="ij")
        self.X_sign = self.sign(X, self.img_shape[-2]).unsqueeze(0).repeat(batch_size, 1, 1)
        self.Y_sign = self.sign(Y, self.img_shape[-1]).unsqueeze(0).repeat(batch_size, 1, 1)

        x = torch.fft.ifft2(
                self.transformation_mask(img,batch_size).unsqueeze(1).repeat(1, self.img_shape[-3], 1, 1) * torch.fft.fft2(img)
        ) if self.img_3D else torch.fft.ifft2(
                self.transformation_mask(img,batch_size) * torch.fft.fft2(img))
        return torch.real(x)



    @staticmethod
    def sign(Z, n):
        return ((0 < Z) & (Z <= n / 2 - 1)).float() + -((n / 2 + 1 <= Z) & (Z <= n - 1)).float()

    def transformation_mask(self, img, batch_size):
        if self.img_3D:
            vector = ImageTransformer(in_channels=self.img_shape[-3], output_size=15,img_3D=self.img_3D)(img)
        else:
            vector = ImageTransformer(in_channels=self.img_shape[-2], output_size=15,img_3D=None)(img)
        v_1 = torch.where(vector[:, :7] > 0, 1, -1)
        v_2 = WeightedThreshold(8)(vector[:, 7:])
        v_3 = v_1 * v_2[:,:-1]

        return (((v_3[:, 0] + v_3[:, 1])* 0.5) + 1j * (
                (v_3[:, 2] + v_3[:, 3])* 0.5)).view(batch_size, 1, 1) * \
            ((v_3[:, 4]).view(batch_size, 1, 1) * self.X_sign + (v_3[:, 5]).view(
                batch_size, 1, 1) * self.Y_sign + (v_3[:, 6]).view(batch_size, 1,
                                                                                      1) * self.X_sign * self.Y_sign + v_2[
                                                                                                                       :,
                                                                                                                       7].view(
                batch_size, 1, 1))


class WeightedThreshold(torch.nn.Module):
    def __init__(self, input_shape):
        super(WeightedThreshold, self).__init__()
        self.thresholds = torch.nn.Parameter(torch.rand(input_shape,dtype=torch.float32))

    def forward(self, x):
        # Use self.thresholds for element-wise thresholding
        y = (x > self.thresholds).float()
        return y


class ImageTransformer(torch.nn.Module):
    def __init__(self, in_channels, output_size=15, img_3D=None,*args, **kwargs):
        super(ImageTransformer, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        if img_3D:
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
                torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                torch.nn.Conv2d(32, 32, kernel_size=2, padding=1),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(32, 16, kernel_size=2, stride=2),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
            )
        else:
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv1d(self.in_channels, 32, kernel_size=3, padding=1),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                torch.nn.Conv1d(32, 32, kernel_size=3, padding=1),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                torch.nn.Conv1d(32, 16, kernel_size=2, stride=2),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
            )

        self.output_size = output_size
        self.fc = None
    def build_fully_connected_layers(self,input):
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input.size(-1), 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        if self.fc is None:
            self.build_fully_connected_layers(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    input_data = torch.randn(64,16, 256, 256)
    hilbert_model = Hilbert2D(transformation="CC", mask_type="HT", axis=0)
    output = hilbert_model(input_data)
    # print(output.size())
    #
    # print(ImageTransformer(3, output_size=15)(input_data).size())
    print("goes to the next layer")
    input_data = torch.randn(64, 16, 256, 256)
    model =  WeightedHilbert2D()
    model(input_data)
    print("Done 1")
    input_data = torch.randn(32, 16, 256, 256)
    model(input_data)
    print("Done 2")
