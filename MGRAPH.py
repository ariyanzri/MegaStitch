import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import subprocess
import os

# from non_linear_optimizer import Non_Linear_Reprojection_Method


class Graph:
    def __init__(self, nodes, nodes_name_to_index_dict, edge_weight_matrix, pos):

        if nodes is None:
            self.initialize_test()

        else:
            self.nodes = nodes
            self.nodes_name_to_index_dict = nodes_name_to_index_dict
            self.edge_weight_matrix = edge_weight_matrix

            self.node_count = len(self.nodes)
            self.node_positions = pos

    def initialize_test(self):
        n = 15
        self.nodes = ["a_{0}".format(i) for i in range(n)]
        self.nodes_name_to_index_dict = {a: i for i, a in enumerate(self.nodes)}
        self.edge_weight_matrix = np.random.rand(n, n)
        self.edge_weight_matrix = np.matmul(
            self.edge_weight_matrix.T, self.edge_weight_matrix
        )

        self.node_count = len(self.nodes)
        self.node_positions = None

    def draw_graph(self):

        if self.node_positions is None:
            g = nx.DiGraph()

            for node in self.nodes:
                for neighbor_node in self.nodes:

                    if type(node) == str:
                        i = self.nodes_name_to_index_dict[node]
                        j = self.nodes_name_to_index_dict[neighbor_node]

                    w = self.edge_weight_matrix[i][j]

                    if w != 0:
                        g.add_weighted_edges_from([(i, j, w)])

            nx.draw(g)
            plt.show()

        else:

            g = nx.DiGraph()

            for node in self.nodes:
                i = self.nodes_name_to_index_dict[node]
                x = self.node_positions[i][0]
                y = self.node_positions[i][1]

                g.add_node(i, pos=(x, y))

            for node in self.nodes:
                for neighbor_node in self.nodes:
                    i = self.nodes_name_to_index_dict[node]
                    j = self.nodes_name_to_index_dict[neighbor_node]
                    w = self.edge_weight_matrix[i][j]

                    if w != 0:
                        g.add_edge(i, j, weight=w)

            pos = nx.get_node_attributes(g, "pos")
            nx.draw(g, pos)
            labels = nx.get_edge_attributes(g, "weight")
            nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

            plt.show()

    def find_min_key(self, keys, mstSet):
        min_value = sys.maxsize

        for v in range(self.node_count):
            if keys[v] < min_value and mstSet[v] == False:
                min_value = keys[v]
                min_index = v

        return min_index

    def generate_MST_prim(self, starting_vertex):

        keys = [sys.maxsize] * self.node_count
        parents = [None] * self.node_count
        mstSet = [False] * self.node_count

        keys[starting_vertex] = 0
        parents[starting_vertex] = -1

        for count in range(self.node_count):
            u = self.find_min_key(keys, mstSet)
            mstSet[u] = True

            for v in range(self.node_count):

                if (
                    self.edge_weight_matrix[u][v] > 0
                    and mstSet[v] == False
                    and keys[v] > self.edge_weight_matrix[u][v]
                ):
                    keys[v] = self.edge_weight_matrix[u][v]
                    parents[v] = u

        new_edges = np.zeros((self.node_count, self.node_count))

        queue_traverse = []

        for v, p in enumerate(parents):
            if p == -1:
                queue_traverse = [v]
                break

        while len(queue_traverse) > 0:
            u = queue_traverse.pop()

            for v, p in enumerate(parents):
                if p == u:
                    queue_traverse = [v] + queue_traverse

                    new_edges[p][v] = self.edge_weight_matrix[p][v]
                    new_edges[v][p] = self.edge_weight_matrix[v][p]

        g = Graph(
            self.nodes, self.nodes_name_to_index_dict, new_edges, self.node_positions
        )

        return g


class Ceres_CPP:
    def __init__(
        self,
        images,
        images_dict,
        image_name_to_index_dict,
        image_index_to_name_dict,
        pairwise_homography_dict,
        absolute_homography_dict,
        temp_path,
        max_matches,
    ):

        self.images = images
        self.images_dict = images_dict
        self.image_name_to_index_dict = image_name_to_index_dict
        self.image_index_to_name_dict = image_index_to_name_dict
        self.pairwise_homography_dict = pairwise_homography_dict
        self.absolute_homography_dict = absolute_homography_dict
        self.temp_path = temp_path
        self.max_matches = max_matches

    def save_to_file(self):

        file_content = "{0}\n".format(len(self.images))

        for img_name in self.absolute_homography_dict:
            i = self.image_name_to_index_dict[img_name]
            h = self.absolute_homography_dict[img_name]

            h = h.reshape(9)
            file_content += "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(
                i, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]
            )

        total_pairs = sum(
            [
                len(self.pairwise_homography_dict[img])
                for img in self.pairwise_homography_dict
            ]
        )

        file_content += "{0}\n".format(total_pairs)

        for img1 in self.pairwise_homography_dict:
            for img2 in self.pairwise_homography_dict[img1]:

                matches = self.pairwise_homography_dict[img1][img2][1]
                i = self.image_name_to_index_dict[img1]
                j = self.image_name_to_index_dict[img2]

                matche_count = min(self.max_matches, len(matches))

                file_content += "{0} {1} {2} ".format(i, j, matche_count)

                for m in matches[:matche_count]:

                    kp1 = self.images_dict[img1].kp[m.trainIdx]
                    kp2 = self.images_dict[img2].kp[m.queryIdx]

                    p1 = (kp1.pt[0], kp1.pt[1])
                    p2 = (kp2.pt[0], kp2.pt[1])

                    file_content += "{0} {1} {2} {3} ".format(
                        p1[0], p1[1], p2[0], p2[1]
                    )

                file_content = file_content[:-1]
                file_content += "\n"

        f = open(self.temp_path, "w+")
        f.write(file_content)
        f.close()

    def load_from_file(self):

        f = open(self.temp_path, "r")
        file_content = f.read()
        f.close()

        os.remove(self.temp_path)

        new_absolute_homography_dict = {}

        lines = file_content.split("\n")
        num = int(lines[0])
        lines = lines[1:]

        for line in lines:
            if line == "":
                continue

            elements = line.split()
            image_index = int(elements[0])
            h = np.zeros(9)

            for i, e in enumerate(elements[1:]):
                h[i] = float(e)

            h = h.reshape((3, 3))

            new_absolute_homography_dict[self.image_index_to_name_dict[image_index]] = h

            num -= 1
            if num == 0:
                break

        return new_absolute_homography_dict


class MGRAPH:
    def __init__(
        self,
        images,
        pairwise_homography_dict,
        image_name_to_index_dict,
        image_locations,
        reference_image,
        transformation_type,
        use_ceres,
        mx_nmb_in,
    ):

        self.images = images

        self.images_dict = {}

        for img in self.images:
            self.images_dict[img.name] = img

        self.pairwise_homography_dict = pairwise_homography_dict
        self.image_name_to_index_dict = image_name_to_index_dict
        self.image_locations = image_locations
        self.reference_image = reference_image

        self.image_index_to_name_dict = {}

        for name in self.image_name_to_index_dict:
            self.image_index_to_name_dict[self.image_name_to_index_dict[name]] = name

        self.edge_matrix = np.zeros((len(self.images), len(self.images)))

        for img_name in self.pairwise_homography_dict:
            for neighbor_name in self.pairwise_homography_dict[img_name]:

                i = self.image_name_to_index_dict[img_name]
                j = self.image_name_to_index_dict[neighbor_name]

                self.edge_matrix[i][j] = self.pairwise_homography_dict[img_name][
                    neighbor_name
                ][2]

                if (
                    neighbor_name in self.pairwise_homography_dict
                    and img_name in self.pairwise_homography_dict[neighbor_name]
                ):
                    if self.edge_matrix[j][i] > self.edge_matrix[i][j]:
                        self.edge_matrix[i][j] = self.edge_matrix[j][i]
                    else:
                        self.edge_matrix[j][i] = self.edge_matrix[i][j]

        locations = np.zeros((len(self.images), 2))

        if self.image_locations is not None:
            for image in self.images:
                i = self.image_name_to_index_dict[image.name]

                locations[i] = np.array(self.image_locations[image.name])

        self.underlying_graph = Graph(
            [img.name for img in self.images],
            self.image_name_to_index_dict,
            self.edge_matrix,
            locations,
        )

        self.MST = self.underlying_graph.generate_MST_prim(
            self.image_name_to_index_dict[self.reference_image.name]
        )

        self.absolute_homography_dict = self.get_absolute_homographies()

        self.transformation_type = transformation_type

        self.use_ceres = use_ceres

        self.max_number_inliers = mx_nmb_in

        print(
            ">>> MGRAPH initialized and absolute homographies calculated successfully."
        )

    def get_absolute_homographies(self):

        absolute_homography_dict = {}

        queue_traverse = [self.reference_image.name]

        absolute_homography_dict[self.reference_image.name] = np.eye(3)

        while len(queue_traverse) > 0:
            u = queue_traverse.pop()

            for v, edge in enumerate(
                self.MST.edge_weight_matrix[self.image_name_to_index_dict[u]]
            ):

                v_name = self.image_index_to_name_dict[v]

                if v_name in absolute_homography_dict:
                    continue

                if edge != 0:

                    absolute_u = absolute_homography_dict[u]
                    H = np.matmul(
                        absolute_u, self.pairwise_homography_dict[u][v_name][0]
                    )
                    absolute_homography_dict[v_name] = H

                    queue_traverse = [v_name] + queue_traverse

        return absolute_homography_dict

    def optimize(self):

        if self.use_ceres:
            cpp = Ceres_CPP(
                self.images,
                self.images_dict,
                self.image_name_to_index_dict,
                self.image_index_to_name_dict,
                self.pairwise_homography_dict,
                self.absolute_homography_dict,
                "/home/ariyan/Desktop/tmp.txt",
                20,
            )
            cpp.save_to_file()

            command = "./cpp/homography_global_optimization"

            process = subprocess.Popen([command, "/home/ariyan/Desktop/tmp.txt"])
            process.wait()

            self.absolute_homography_dict = cpp.load_from_file()

        else:

            solver = Non_Linear_Reprojection_Method(
                self.absolute_homography_dict,
                self.pairwise_homography_dict,
                self.image_name_to_index_dict,
                self.max_number_inliers,
                self.images_dict,
                self.transformation_type,
                self.reference_image.name,
            )
            self.absolute_homography_dict = solver.solve()

        return self.absolute_homography_dict


if __name__ == "__main__":

    g = Graph(None, None, None, None)

    g.draw_graph()
    new_g = g.generate_MST_prim(0)

    new_g.draw_graph()
