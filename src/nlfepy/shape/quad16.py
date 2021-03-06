import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Quad16(ShapeFunction):
    """
    Quad16 (16-node quadrilateral element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "QUAD"
        self._name = "QUAD16"
        self._n_dof = 2
        self._n_node = 16
        self._n_intgp = 16
        self._n_face = 4
        self._n_fnode = 4
        self._weight = np.array(
            [
                0.121002993285602,
                0.226851851851852,
                0.226851851851852,
                0.121002993285602,
                0.226851851851852,
                0.425293303010694,
                0.425293303010694,
                0.226851851851852,
                0.226851851851852,
                0.425293303010694,
                0.425293303010694,
                0.226851851851852,
                0.121002993285602,
                0.226851851851852,
                0.226851851851852,
                0.121002993285602,
            ]
        )
        self._Shpfnc = np.array(
            [
                [
                    0.435607477928195,
                    0.002226685158092,
                    0.001096772541183,
                    0.032501682281677,
                    0.002226685158092,
                    0.000011382097518,
                    0.000005606348061,
                    0.000166138134023,
                    0.001096772541183,
                    0.000005606348061,
                    0.000002761453988,
                    0.00008183273813,
                    0.032501682281677,
                    0.000166138134023,
                    0.00008183273813,
                    0.002425025750621,
                ],
                [
                    0.032501682281677,
                    0.001096772541183,
                    0.002226685158092,
                    0.435607477928195,
                    0.000166138134023,
                    0.000005606348061,
                    0.000011382097518,
                    0.002226685158092,
                    0.00008183273813,
                    0.000002761453988,
                    0.000005606348061,
                    0.001096772541183,
                    0.002425025750621,
                    0.00008183273813,
                    0.000166138134023,
                    0.032501682281677,
                ],
                [
                    0.002425025750621,
                    0.00008183273813,
                    0.000166138134023,
                    0.032501682281677,
                    0.00008183273813,
                    0.000002761453988,
                    0.000005606348061,
                    0.001096772541183,
                    0.000166138134023,
                    0.000005606348061,
                    0.000011382097518,
                    0.002226685158092,
                    0.032501682281677,
                    0.001096772541183,
                    0.002226685158092,
                    0.435607477928195,
                ],
                [
                    0.032501682281677,
                    0.000166138134023,
                    0.00008183273813,
                    0.002425025750621,
                    0.001096772541183,
                    0.000005606348061,
                    0.000002761453988,
                    0.00008183273813,
                    0.002226685158092,
                    0.000011382097518,
                    0.000005606348061,
                    0.000166138134023,
                    0.435607477928195,
                    0.002226685158092,
                    0.001096772541183,
                    0.032501682281677,
                ],
                [
                    0.343821825039651,
                    0.663230356936454,
                    -0.006548149562924,
                    -0.151925320176718,
                    0.001757506456237,
                    0.003390220019203,
                    -0.000033472032009,
                    -0.000776593315581,
                    0.000865674617333,
                    0.001669881443327,
                    -0.000016486931469,
                    -0.000382517582739,
                    0.025653342252281,
                    0.049485152190764,
                    -0.000488572596686,
                    -0.011335499818347,
                ],
                [
                    -0.151925320176718,
                    -0.006548149562924,
                    0.663230356936454,
                    0.343821825039651,
                    -0.000776593315581,
                    -0.000033472032009,
                    0.003390220019203,
                    0.001757506456237,
                    -0.000382517582739,
                    -0.000016486931469,
                    0.001669881443327,
                    0.000865674617333,
                    -0.011335499818347,
                    -0.000488572596686,
                    0.049485152190764,
                    0.025653342252281,
                ],
                [
                    0.025653342252281,
                    0.000865674617333,
                    0.001757506456237,
                    0.343821825039651,
                    0.049485152190764,
                    0.001669881443327,
                    0.003390220019203,
                    0.663230356936454,
                    -0.000488572596686,
                    -0.000016486931469,
                    -0.000033472032009,
                    -0.006548149562924,
                    -0.011335499818347,
                    -0.000382517582739,
                    -0.000776593315581,
                    -0.151925320176718,
                ],
                [
                    -0.011335499818347,
                    -0.000382517582739,
                    -0.000776593315581,
                    -0.151925320176718,
                    -0.000488572596686,
                    -0.000016486931469,
                    -0.000033472032009,
                    -0.006548149562924,
                    0.049485152190764,
                    0.001669881443327,
                    0.003390220019203,
                    0.663230356936454,
                    0.025653342252281,
                    0.000865674617333,
                    0.001757506456237,
                    0.343821825039651,
                ],
                [
                    -0.011335499818347,
                    -0.000488572596686,
                    0.049485152190764,
                    0.025653342252281,
                    -0.000382517582739,
                    -0.000016486931469,
                    0.001669881443327,
                    0.000865674617333,
                    -0.000776593315581,
                    -0.000033472032009,
                    0.003390220019203,
                    0.001757506456237,
                    -0.151925320176718,
                    -0.006548149562924,
                    0.663230356936454,
                    0.343821825039651,
                ],
                [
                    0.025653342252281,
                    0.049485152190764,
                    -0.000488572596686,
                    -0.011335499818347,
                    0.000865674617333,
                    0.001669881443327,
                    -0.000016486931469,
                    -0.000382517582739,
                    0.001757506456237,
                    0.003390220019203,
                    -0.000033472032009,
                    -0.000776593315581,
                    0.343821825039651,
                    0.663230356936454,
                    -0.006548149562924,
                    -0.151925320176718,
                ],
                [
                    -0.151925320176718,
                    -0.000776593315581,
                    -0.000382517582739,
                    -0.011335499818347,
                    -0.006548149562924,
                    -0.000033472032009,
                    -0.000016486931469,
                    -0.000488572596686,
                    0.663230356936454,
                    0.003390220019203,
                    0.001669881443327,
                    0.049485152190764,
                    0.343821825039651,
                    0.001757506456237,
                    0.000865674617333,
                    0.025653342252281,
                ],
                [
                    0.343821825039651,
                    0.001757506456237,
                    0.000865674617333,
                    0.025653342252281,
                    0.663230356936454,
                    0.003390220019203,
                    0.001669881443327,
                    0.049485152190764,
                    -0.006548149562924,
                    -0.000033472032009,
                    -0.000016486931469,
                    -0.000488572596686,
                    -0.151925320176718,
                    -0.000776593315581,
                    -0.000382517582739,
                    -0.011335499818347,
                ],
                [
                    0.271376074478415,
                    0.523482913627069,
                    -0.005168406988937,
                    -0.119913554058645,
                    0.523482913627069,
                    1.009795581228668,
                    -0.009969827865552,
                    -0.231312567928641,
                    -0.005168406988937,
                    -0.009969827865552,
                    0.000098433256707,
                    0.002283775576223,
                    -0.119913554058645,
                    -0.231312567928641,
                    0.002283775576223,
                    0.052986470802971,
                ],
                [
                    -0.119913554058645,
                    -0.005168406988937,
                    0.523482913627069,
                    0.271376074478415,
                    -0.231312567928641,
                    -0.009969827865552,
                    1.009795581228668,
                    0.523482913627069,
                    0.002283775576223,
                    0.000098433256707,
                    -0.009969827865552,
                    -0.005168406988937,
                    0.052986470802971,
                    0.002283775576223,
                    -0.231312567928641,
                    -0.119913554058645,
                ],
                [
                    0.052986470802971,
                    0.002283775576223,
                    -0.231312567928641,
                    -0.119913554058645,
                    0.002283775576223,
                    0.000098433256707,
                    -0.009969827865552,
                    -0.005168406988937,
                    -0.231312567928641,
                    -0.009969827865552,
                    1.009795581228668,
                    0.523482913627069,
                    -0.119913554058645,
                    -0.005168406988937,
                    0.523482913627069,
                    0.271376074478415,
                ],
                [
                    -0.119913554058645,
                    -0.231312567928641,
                    0.002283775576223,
                    0.052986470802971,
                    -0.005168406988937,
                    -0.009969827865552,
                    0.000098433256707,
                    0.002283775576223,
                    0.523482913627069,
                    1.009795581228668,
                    -0.009969827865552,
                    -0.231312567928641,
                    0.271376074478415,
                    0.523482913627069,
                    -0.005168406988937,
                    -0.119913554058645,
                ],
            ]
        )
        self._Bmatrix_nat = np.array(
            [
                [
                    [
                        -1.424063648007379,
                        -0.339923986314303,
                        0.164952196948277,
                        -0.145265248890436,
                        -0.007279354808779,
                        -0.001737581959808,
                        0.000843182514879,
                        -0.000742549175761,
                        -0.003585507561669,
                        -0.000855860638693,
                        0.000415316654069,
                        -0.000365748854746,
                        -0.106252685230648,
                        -0.025362515482183,
                        0.012307465249164,
                        -0.010838576482803,
                    ],
                    [
                        0.145265248890436,
                        -0.164952196948277,
                        0.339923986314303,
                        1.424063648007379,
                        0.000742549175761,
                        -0.000843182514879,
                        0.001737581959808,
                        0.007279354808779,
                        0.000365748854746,
                        -0.000415316654069,
                        0.000855860638693,
                        0.003585507561669,
                        0.010838576482803,
                        -0.012307465249164,
                        0.025362515482183,
                        0.106252685230648,
                    ],
                    [
                        0.010838576482803,
                        -0.012307465249164,
                        0.025362515482183,
                        0.106252685230648,
                        0.000365748854746,
                        -0.000415316654069,
                        0.000855860638693,
                        0.003585507561669,
                        0.000742549175761,
                        -0.000843182514879,
                        0.001737581959808,
                        0.007279354808779,
                        0.145265248890436,
                        -0.164952196948277,
                        0.339923986314303,
                        1.424063648007379,
                    ],
                    [
                        -0.106252685230648,
                        -0.025362515482183,
                        0.012307465249164,
                        -0.010838576482803,
                        -0.003585507561669,
                        -0.000855860638693,
                        0.000415316654069,
                        -0.000365748854746,
                        -0.007279354808779,
                        -0.001737581959808,
                        0.000843182514879,
                        -0.000742549175761,
                        -1.424063648007379,
                        -0.339923986314303,
                        0.164952196948277,
                        -0.145265248890436,
                    ],
                    [
                        2.003384047295989,
                        -0.475112721928877,
                        -0.979988905191458,
                        0.724585648179045,
                        0.010240654144161,
                        -0.002428623244421,
                        -0.005009387719108,
                        0.003703848511143,
                        0.005044120507224,
                        -0.001196238847543,
                        -0.002467416140305,
                        0.001824361800301,
                        0.149477121244752,
                        -0.035449259984146,
                        -0.073119240715494,
                        0.054063012496907,
                    ],
                    [
                        -0.724585648179045,
                        0.979988905191458,
                        0.475112721928877,
                        -2.003384047295989,
                        -0.003703848511143,
                        0.005009387719108,
                        0.002428623244421,
                        -0.010240654144161,
                        -0.001824361800301,
                        0.002467416140305,
                        0.001196238847543,
                        -0.005044120507224,
                        -0.054063012496907,
                        0.073119240715494,
                        0.035449259984146,
                        -0.149477121244752,
                    ],
                    [
                        0.114656808064672,
                        -0.130195573475452,
                        0.268299538624087,
                        1.12400311573905,
                        0.22117233462174,
                        -0.251146525261193,
                        0.517548293355194,
                        2.168195656466325,
                        -0.002183659887002,
                        0.002479598517196,
                        -0.005109813800713,
                        -0.021406844984909,
                        -0.050663660672667,
                        0.057529809759949,
                        -0.118554118267592,
                        -0.496665775125213,
                    ],
                    [
                        -0.050663660672667,
                        0.057529809759949,
                        -0.118554118267592,
                        -0.496665775125213,
                        -0.002183659887002,
                        0.002479598517196,
                        -0.005109813800713,
                        -0.021406844984909,
                        0.22117233462174,
                        -0.251146525261193,
                        0.517548293355194,
                        2.168195656466325,
                        0.114656808064672,
                        -0.130195573475452,
                        0.268299538624087,
                        1.12400311573905,
                    ],
                    [
                        -0.054063012496907,
                        0.073119240715494,
                        0.035449259984146,
                        -0.149477121244752,
                        -0.001824361800301,
                        0.002467416140305,
                        0.001196238847543,
                        -0.005044120507224,
                        -0.003703848511143,
                        0.005009387719108,
                        0.002428623244421,
                        -0.010240654144161,
                        -0.724585648179045,
                        0.979988905191458,
                        0.475112721928877,
                        -2.003384047295989,
                    ],
                    [
                        0.149477121244752,
                        -0.035449259984146,
                        -0.073119240715494,
                        0.054063012496907,
                        0.005044120507224,
                        -0.001196238847543,
                        -0.002467416140305,
                        0.001824361800301,
                        0.010240654144161,
                        -0.002428623244421,
                        -0.005009387719108,
                        0.003703848511143,
                        2.003384047295989,
                        -0.475112721928877,
                        -0.979988905191458,
                        0.724585648179045,
                    ],
                    [
                        0.496665775125213,
                        0.118554118267592,
                        -0.057529809759949,
                        0.050663660672667,
                        0.021406844984909,
                        0.005109813800713,
                        -0.002479598517196,
                        0.002183659887002,
                        -2.168195656466325,
                        -0.517548293355194,
                        0.251146525261193,
                        -0.22117233462174,
                        -1.12400311573905,
                        -0.268299538624087,
                        0.130195573475452,
                        -0.114656808064672,
                    ],
                    [
                        -1.12400311573905,
                        -0.268299538624087,
                        0.130195573475452,
                        -0.114656808064672,
                        -2.168195656466325,
                        -0.517548293355194,
                        0.251146525261193,
                        -0.22117233462174,
                        0.021406844984909,
                        0.005109813800713,
                        -0.002479598517196,
                        0.002183659887002,
                        0.496665775125213,
                        0.118554118267592,
                        -0.057529809759949,
                        0.050663660672667,
                    ],
                    [
                        1.581256507975218,
                        -0.37500302779483,
                        -0.77349813989437,
                        0.57191020030084,
                        3.050234865315921,
                        -0.723378720789274,
                        -1.492073539405661,
                        1.103211543471336,
                        -0.030115319498333,
                        0.007142001274255,
                        0.014731413592165,
                        -0.010892134400427,
                        -0.698713356046986,
                        0.165703428100872,
                        0.341787356128413,
                        -0.252711241594439,
                    ],
                    [
                        -0.57191020030084,
                        0.77349813989437,
                        0.37500302779483,
                        -1.581256507975218,
                        -1.103211543471336,
                        1.492073539405661,
                        0.723378720789274,
                        -3.050234865315921,
                        0.010892134400427,
                        -0.014731413592165,
                        -0.007142001274255,
                        0.030115319498333,
                        0.252711241594439,
                        -0.341787356128413,
                        -0.165703428100872,
                        0.698713356046986,
                    ],
                    [
                        0.252711241594439,
                        -0.341787356128413,
                        -0.165703428100872,
                        0.698713356046986,
                        0.010892134400427,
                        -0.014731413592165,
                        -0.007142001274255,
                        0.030115319498333,
                        -1.103211543471336,
                        1.492073539405661,
                        0.723378720789274,
                        -3.050234865315921,
                        -0.57191020030084,
                        0.77349813989437,
                        0.37500302779483,
                        -1.581256507975218,
                    ],
                    [
                        -0.698713356046986,
                        0.165703428100872,
                        0.341787356128413,
                        -0.252711241594439,
                        -0.030115319498333,
                        0.007142001274255,
                        0.014731413592165,
                        -0.010892134400427,
                        3.050234865315921,
                        -0.723378720789274,
                        -1.492073539405661,
                        1.103211543471336,
                        1.581256507975218,
                        -0.37500302779483,
                        -0.77349813989437,
                        0.57191020030084,
                    ],
                ],
                [
                    [
                        -1.424063648007379,
                        -0.007279354808779,
                        -0.003585507561669,
                        -0.106252685230648,
                        -0.339923986314303,
                        -0.001737581959808,
                        -0.000855860638693,
                        -0.025362515482183,
                        0.164952196948277,
                        0.000843182514879,
                        0.000415316654069,
                        0.012307465249164,
                        -0.145265248890436,
                        -0.000742549175761,
                        -0.000365748854746,
                        -0.010838576482803,
                    ],
                    [
                        -0.106252685230648,
                        -0.003585507561669,
                        -0.007279354808779,
                        -1.424063648007379,
                        -0.025362515482183,
                        -0.000855860638693,
                        -0.001737581959808,
                        -0.339923986314303,
                        0.012307465249164,
                        0.000415316654069,
                        0.000843182514879,
                        0.164952196948277,
                        -0.010838576482803,
                        -0.000365748854746,
                        -0.000742549175761,
                        -0.145265248890436,
                    ],
                    [
                        0.010838576482803,
                        0.000365748854746,
                        0.000742549175761,
                        0.145265248890436,
                        -0.012307465249164,
                        -0.000415316654069,
                        -0.000843182514879,
                        -0.164952196948277,
                        0.025362515482183,
                        0.000855860638693,
                        0.001737581959808,
                        0.339923986314303,
                        0.106252685230648,
                        0.003585507561669,
                        0.007279354808779,
                        1.424063648007379,
                    ],
                    [
                        0.145265248890436,
                        0.000742549175761,
                        0.000365748854746,
                        0.010838576482803,
                        -0.164952196948277,
                        -0.000843182514879,
                        -0.000415316654069,
                        -0.012307465249164,
                        0.339923986314303,
                        0.001737581959808,
                        0.000855860638693,
                        0.025362515482183,
                        1.424063648007379,
                        0.007279354808779,
                        0.003585507561669,
                        0.106252685230648,
                    ],
                    [
                        -1.12400311573905,
                        -2.168195656466325,
                        0.021406844984909,
                        0.496665775125213,
                        -0.268299538624087,
                        -0.517548293355194,
                        0.005109813800713,
                        0.118554118267592,
                        0.130195573475452,
                        0.251146525261193,
                        -0.002479598517196,
                        -0.057529809759949,
                        -0.114656808064672,
                        -0.22117233462174,
                        0.002183659887002,
                        0.050663660672667,
                    ],
                    [
                        0.496665775125213,
                        0.021406844984909,
                        -2.168195656466325,
                        -1.12400311573905,
                        0.118554118267592,
                        0.005109813800713,
                        -0.517548293355194,
                        -0.268299538624087,
                        -0.057529809759949,
                        -0.002479598517196,
                        0.251146525261193,
                        0.130195573475452,
                        0.050663660672667,
                        0.002183659887002,
                        -0.22117233462174,
                        -0.114656808064672,
                    ],
                    [
                        0.149477121244752,
                        0.005044120507224,
                        0.010240654144161,
                        2.003384047295989,
                        -0.035449259984146,
                        -0.001196238847543,
                        -0.002428623244421,
                        -0.475112721928877,
                        -0.073119240715494,
                        -0.002467416140305,
                        -0.005009387719108,
                        -0.979988905191458,
                        0.054063012496907,
                        0.001824361800301,
                        0.003703848511143,
                        0.724585648179045,
                    ],
                    [
                        -0.054063012496907,
                        -0.001824361800301,
                        -0.003703848511143,
                        -0.724585648179045,
                        0.073119240715494,
                        0.002467416140305,
                        0.005009387719108,
                        0.979988905191458,
                        0.035449259984146,
                        0.001196238847543,
                        0.002428623244421,
                        0.475112721928877,
                        -0.149477121244752,
                        -0.005044120507224,
                        -0.010240654144161,
                        -2.003384047295989,
                    ],
                    [
                        -0.050663660672667,
                        -0.002183659887002,
                        0.22117233462174,
                        0.114656808064672,
                        0.057529809759949,
                        0.002479598517196,
                        -0.251146525261193,
                        -0.130195573475452,
                        -0.118554118267592,
                        -0.005109813800713,
                        0.517548293355194,
                        0.268299538624087,
                        -0.496665775125213,
                        -0.021406844984909,
                        2.168195656466325,
                        1.12400311573905,
                    ],
                    [
                        0.114656808064672,
                        0.22117233462174,
                        -0.002183659887002,
                        -0.050663660672667,
                        -0.130195573475452,
                        -0.251146525261193,
                        0.002479598517196,
                        0.057529809759949,
                        0.268299538624087,
                        0.517548293355194,
                        -0.005109813800713,
                        -0.118554118267592,
                        1.12400311573905,
                        2.168195656466325,
                        -0.021406844984909,
                        -0.496665775125213,
                    ],
                    [
                        -0.724585648179045,
                        -0.003703848511143,
                        -0.001824361800301,
                        -0.054063012496907,
                        0.979988905191458,
                        0.005009387719108,
                        0.002467416140305,
                        0.073119240715494,
                        0.475112721928877,
                        0.002428623244421,
                        0.001196238847543,
                        0.035449259984146,
                        -2.003384047295989,
                        -0.010240654144161,
                        -0.005044120507224,
                        -0.149477121244752,
                    ],
                    [
                        2.003384047295989,
                        0.010240654144161,
                        0.005044120507224,
                        0.149477121244752,
                        -0.475112721928877,
                        -0.002428623244421,
                        -0.001196238847543,
                        -0.035449259984146,
                        -0.979988905191458,
                        -0.005009387719108,
                        -0.002467416140305,
                        -0.073119240715494,
                        0.724585648179045,
                        0.003703848511143,
                        0.001824361800301,
                        0.054063012496907,
                    ],
                    [
                        1.581256507975218,
                        3.050234865315921,
                        -0.030115319498333,
                        -0.698713356046986,
                        -0.37500302779483,
                        -0.723378720789274,
                        0.007142001274255,
                        0.165703428100872,
                        -0.77349813989437,
                        -1.492073539405661,
                        0.014731413592165,
                        0.341787356128413,
                        0.57191020030084,
                        1.103211543471336,
                        -0.010892134400427,
                        -0.252711241594439,
                    ],
                    [
                        -0.698713356046986,
                        -0.030115319498333,
                        3.050234865315921,
                        1.581256507975218,
                        0.165703428100872,
                        0.007142001274255,
                        -0.723378720789274,
                        -0.37500302779483,
                        0.341787356128413,
                        0.014731413592165,
                        -1.492073539405661,
                        -0.77349813989437,
                        -0.252711241594439,
                        -0.010892134400427,
                        1.103211543471336,
                        0.57191020030084,
                    ],
                    [
                        0.252711241594439,
                        0.010892134400427,
                        -1.103211543471336,
                        -0.57191020030084,
                        -0.341787356128413,
                        -0.014731413592165,
                        1.492073539405661,
                        0.77349813989437,
                        -0.165703428100872,
                        -0.007142001274255,
                        0.723378720789274,
                        0.37500302779483,
                        0.698713356046986,
                        0.030115319498333,
                        -3.050234865315921,
                        -1.581256507975218,
                    ],
                    [
                        -0.57191020030084,
                        -1.103211543471336,
                        0.010892134400427,
                        0.252711241594439,
                        0.77349813989437,
                        1.492073539405661,
                        -0.014731413592165,
                        -0.341787356128413,
                        0.37500302779483,
                        0.723378720789274,
                        -0.007142001274255,
                        -0.165703428100872,
                        -1.581256507975218,
                        -3.050234865315921,
                        0.030115319498333,
                        0.698713356046986,
                    ],
                ],
            ]
        )
        self._idx_face = np.array(
            [[0, 1, 4, 5], [1, 2, 6, 7], [2, 3, 8, 9], [3, 0, 10, 11]]
        )
