import numpy as np
import json

class RuleGroup:
    def __init__(self, rule_name, xml_rule):
        self.rules = {}
        self.rule_name = rule_name
        for rule in xml_rule.childNodes:
            items = rule.attributes.items()
            (a, attr) = items[0]
            (b, value) = items[1]
            self.rules[attr] = value

    def __str__(self):
        return self.rule_name + json.dumps(self.rules, indent=4, sort_keys=True)


class Rules:
    def __init__(self, xml_file):
        from xml.dom import minidom
        self.rules = []

        xmldoc = minidom.parse(str(xml_file))
        xmlRules = xmldoc.getElementsByTagName('Data')[0].getElementsByTagName('Rules')[0].childNodes

        if len(xmlRules) == 1:
            self.rules.append(RuleGroup("", xmlRules[0]))

        else:
            self.rules.append(RuleGroup("First ", xmlRules[0]))
            self.rules.append(RuleGroup("Second ", xmlRules[1]))

    def __str__(self):
        string = ""
        for rule in self.rules:
            string += str(rule) + "\n"
        return string


class Sample:
    def __init__(self, npz_file, xml_file):
        with np.load(npz_file) as data:
            self.questionPanels = data['image'][:8]
            self.answerPanels = data['image'][8:16]
            self.answer = int(data['target']) + 1
        self.rules = Rules(xml_file)

    def display(self):
        import matplotlib.pyplot as plt
        fig, ims = plt.subplots(6, 4)

        # plot
        for i in range(3):
            for j in range(3):
                if 3 * i + j >= 8:
                    ims[i][j].axis('off')
                    break
                image = self.questionPanels[3 * i + j, :].squeeze()
                ims[i][j].imshow(image, cmap='gray')
                plt.setp(ims[i][j].get_yticklabels(), visible=False)
                plt.setp(ims[i][j].get_xticklabels(), visible=False)
                ims[i][j].tick_params(axis='both', which='both', length=0)

            ims[i][3].axis('off')
        for i in range(4):
            ims[3][i].axis('off')

        for i in range(8):
            image = self.answerPanels[i, :].squeeze()
            ims[4 + i // 4][i % 4].imshow(image, cmap='gray')
            ims[4 + i // 4][i % 4].set_ylabel(str(i+1), rotation=0)

            plt.setp(ims[4 + i // 4][i % 4].get_yticklabels(), visible=False)
            plt.setp(ims[4 + i // 4][i % 4].get_xticklabels(), visible=False)
            ims[4 + i // 4][i % 4].tick_params(axis='both', which='both', length=0)
        plt.show()